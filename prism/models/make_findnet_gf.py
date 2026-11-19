#!/usr/bin/env python
"""make_findnet_gf.py —— 从 find_net.py 生成 find_net_gf.py（GF 版）。

把 No-GF 的 FourierUnit 换成 GF 版：FourierUnit 里加可学习 Gaussian 带通
（center/width/mask，逐 transformed-channel），频域数据先乘 mask 再卷积。
其余（GroupNorm、SpectralTransform/FFC/ProxNet/FINDNet）完全不变。
"""
import re

SRC = "find_net.py"      # 服务器 CWD 下
DST = "find_net_gf.py"

GF_FILTER = '''def create_gaussian_filter(freq_center=0.0, freq_bandwidth=0.2, dimensions=(256, 129),
                            adjust_for_fft=True, add_extra_dim=True):
    """GF 分支的可学习 Gaussian 带通滤波（原版 ffc.py create_gaussian_filter 平铺）。"""

    def format_input(val):
        if not torch.is_tensor(val):
            val = torch.tensor([val] if not isinstance(val, (tuple, list)) else val)
        if val.ndim <= 3:
            val = val.view(val.numel(), 1, 1)
        return val

    freq_center = format_input(freq_center).clamp(0, 1)
    freq_bandwidth = format_input(freq_bandwidth).clamp(min=1e-12, max=2)
    grid_x, grid_y = torch.meshgrid(torch.arange(dimensions[0]),
                                    torch.arange(dimensions[1]), indexing="ij")
    grid_x = torch.repeat_interleave(grid_x.unsqueeze(0), freq_center.shape[0], dim=0).to(freq_center.device)
    grid_y = torch.repeat_interleave(grid_y.unsqueeze(0), freq_center.shape[0], dim=0).to(freq_center.device)
    center_x = (dimensions[0] - 1) // 2
    center_y = 0
    dist_sq = ((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2).float()
    dist_sq /= dist_sq.max()
    filter_response = torch.exp(-((dist_sq - freq_center ** 2) /
                                  (dist_sq.sqrt() * freq_bandwidth + 1e-12)) ** 2)
    if adjust_for_fft:
        filter_response = torch.roll(filter_response, filter_response.shape[-2] // 2 + 1, -2)
    if add_extra_dim:
        filter_response = filter_response.unsqueeze(0)
    return filter_response


'''

GF_FOURIER = '''class FourierUnit(nn.Module):
    """FFC 的频域分支（GF 版：可学习 Gaussian 带通，对应官方 Gaussian_filter=True）。"""

    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()
        self.groups = groups
        transformed_channels = in_channels * 2
        self.conv = nn.Conv2d(
            in_channels=transformed_channels, out_channels=out_channels * 2,
            kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.norm = _group_norm(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)
        self.center = nn.Parameter(
            torch.tensor([0.1 for _ in range(transformed_channels)], dtype=torch.float32),
            requires_grad=True)
        self.width = nn.Parameter(
            torch.tensor([1.0 for _ in range(transformed_channels)], dtype=torch.float32),
            requires_grad=True)

    def forward(self, x):
        batch = x.shape[0]
        fft_dim = (-2, -1)
        freq_data = torch.fft.rfftn(x, dim=fft_dim, norm="ortho")
        freq_data = torch.stack((freq_data.real, freq_data.imag), dim=-1)
        freq_data = freq_data.permute(0, 1, 4, 2, 3).contiguous()
        freq_data = freq_data.view((batch, -1,) + freq_data.size()[3:])
        mask = create_gaussian_filter(self.center, self.width, dimensions=freq_data.shape[2:])
        freq_data = freq_data * mask
        freq_data = self.conv(freq_data)
        freq_data = self.relu(self.norm(freq_data))
        freq_data = freq_data.view((batch, -1, 2,) + freq_data.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()
        freq_data = torch.complex(freq_data[..., 0], freq_data[..., 1])
        output_tensor = torch.fft.irfftn(freq_data, s=x.shape[2:], dim=fft_dim, norm="ortho")
        return output_tensor


'''


def main():
    src = open(SRC, encoding="utf-8").read()
    # 1. 替换 FourierUnit（No-GF → GF）
    new, n = re.subn(r'class FourierUnit\(nn\.Module\):.*?(?=\nclass SpectralTransform)',
                     GF_FOURIER, src, flags=re.S)
    assert n == 1, f"FourierUnit 匹配数 = {n}（应=1）"
    # 2. 在 FourierUnit 前插入 create_gaussian_filter
    new = new.replace("class FourierUnit(nn.Module):", GF_FILTER + "class FourierUnit(nn.Module):", 1)
    # 3. 更新注释
    new = new.replace("Gaussian_filter=False 分支", "Gaussian_filter=True 分支（GF）")
    open(DST, "w", encoding="utf-8").write(new)
    print(f"[ok] {DST} written, 长度 {len(new)}")


if __name__ == "__main__":
    main()
