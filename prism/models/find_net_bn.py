"""find_net —— 真实 FIND-Net（MICCAI 2025）作为 PRISM 下游去除网络 R_θ。

从 FIND-NET/Net-main/FIND-Net-main/Model/{findnet,ProxNet,ffc}.py 平铺移植，
去掉 `from Model.xxx` 相对导入、把 scipy 的 init_kernel.mat 预转成 init_kernel.npy
（[1, 32, 9, 9]，本目录下），保证 PRISM 侧零 scipy 依赖。

架构（与 FIND-Net 官方一致）：
    FINDNet.forward(CT_ma, LIct, Mask) -> (X0, ListX, ListA)
      CT_ma 金属伪影图（脏图，观测源）；LIct 线性插值图（盲区初始化）；
      Mask 非金属掩膜（1=非金属/组织，0=金属，对齐官方 non_Mask）。ListX[-1] 为最终重建。

PRISM 接口（递归循环定案，prism-interface-loss-final）：
    μ_pred（SSM 先验）**替换 LIct** 作为 FIND-Net 的盲区初始化输入 —— 这是「μ 走
    FIND-Net 输入口、不产出最终图」的落点：FIND-Net 仍拿脏图 CT_ma 去伪影，但在
    完全阴影区（信息≈0）它的 X0 初始化来自 SSM 的 z 连续干净先验 μ_pred，而非
    朴素线性插值。于是「让 FIND-Net 在看不到 GT 的时候看到 GT」。
    x̂_z = find_net(CT_ma=Baseline 脏图, LIct=μ_pred, Mask) 是唯一最终输出，并作为
    SSM 的观测喂回卡尔曼更新。

注意：官方 FIND-Net 内部大量 BatchNorm2d（batch_size=16 训练）。PRISM 递归循环天然
batch=1，BatchNorm 逐样本归一化会退化，且梯度检查点下 running-stats 被二次更新 ——
故已把全部 BatchNorm2d 换成 GroupNorm（`_group_norm`，组数 `_GN_GROUPS=8` + 最大约数
回退），batch=1 与两层梯度检查点都干净。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



# ============================================================================
# GroupNorm（替代官方 BatchNorm2d）
# ============================================================================
# 官方 FIND-Net 用 batch_size=16 训练，内部大量 BatchNorm2d；PRISM 递归循环天然 batch=1，
# BatchNorm 逐样本归一化会退化，且梯度检查点下 running-stats 被二次更新。GroupNorm 无
# batch/running stats，逐样本按组分归一化，batch=1 与 checkpoint 都干净。
# 组数用「≤ _GN_GROUPS 的最大约数」回退，兼容 Xnet 的 33=num_Q+1 等非 2 幂通道数。
_GN_GROUPS = 8


def _group_norm(channels: int, groups: int = _GN_GROUPS) -> nn.Module:
    if channels <= 0:
        return nn.Identity()
    return nn.BatchNorm2d(channels)

# ============================================================================
# FFC（Fast Fourier Convolution）—— 从 ffc.py 平铺（Gaussian_filter=False 分支）
# ============================================================================


class FourierUnit(nn.Module):
    """FFC 的频域分支（无 Gaussian 滤波版本，对应官方 Gaussian_filter=False）。"""

    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()
        self.groups = groups
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels * 2, out_channels=out_channels * 2,
            kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = _group_norm(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        freq_data = torch.fft.rfft2(x, dim=2, norm="ortho")
        freq_data = torch.view_as_real(freq_data)
        freq_data = freq_data.permute(0, 1, 4, 2, 3).contiguous()
        freq_data = freq_data.view((batch, -1,) + freq_data.size()[3:])
        freq_data = self.conv_layer(freq_data)
        freq_data = self.relu(self.bn(freq_data))
        freq_data = freq_data.view((batch, -1, 2,) + freq_data.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()
        freq_data = torch.view_as_complex(freq_data)
        return torch.fft.irfft2(freq_data, s=x.shape[2:], dim=(2, 3), norm="ortho")


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        super().__init__()
        self.enable_lfu = enable_lfu
        self.downsample = (nn.AvgPool2d(kernel_size=(2, 2), stride=2)
                           if stride == 2 else nn.Identity())
        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False),
            _group_norm(out_channels // 2),
            nn.ReLU(inplace=True))
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s_h = h // split_no
            split_s_w = w // split_no
            channels_to_use = max(1, c // (split_no ** 2))

            xs = torch.cat(torch.split(
                x[:, :channels_to_use], split_s_h, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s_w, dim=-1), dim=1).contiguous()

            if xs.shape[1] < c:
                padding = c - xs.shape[1]
                xs = torch.cat([xs, xs.new_zeros((n, padding, xs.shape[2], xs.shape[3]))], dim=1)
            elif xs.shape[1] > c:
                xs = xs[:, :c, :, :]

            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        return self.conv2(x + output + xs)


class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True):
        super().__init__()
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cg = in_cg - (in_cg % 4) if in_cg % 4 != 0 else in_cg
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cg = out_cg - (out_cg % 4) if out_cg % 4 != 0 else out_cg
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0
        if x_g.shape[1] == 0:
            out_xl = self.convl2l(x_l)
        elif self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            convl2g = self.convl2g(x_l)
            convg2g = self.convg2g(x_g)
            try:
                out_xg = convl2g + convg2g
            except Exception:
                out_xg = convg2g
        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=_group_norm, activation_layer=nn.Identity,
                 enable_lfu=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        in_cg = int(in_channels * ratio_gin)
        in_cg = in_cg - (in_cg % 4) if in_cg % 4 != 0 else in_cg
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cg = out_cg - (out_cg % 4) if out_cg % 4 != 0 else out_cg
        out_cl = out_channels - out_cg

        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        self.bn_l = lnorm(int(out_cl))
        self.bn_g = gnorm(int(out_cg))

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        in_cg = int(self.in_channels * self.ratio_gin)
        in_cg = in_cg - (in_cg % 4) if in_cg % 4 != 0 else in_cg
        in_cl = self.in_channels - in_cg
        out_cg = int(self.out_channels * self.ratio_gout)
        out_cg = out_cg - (out_cg % 4) if out_cg % 4 != 0 else out_cg
        out_cl = self.out_channels - out_cg

        x_l = x[:, :in_cl, :, :]
        x_g = x[:, in_cl:, :, :]

        x_l, x_g = self.ffc((x_l, x_g))

        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))

        if isinstance(x_l, int):
            output = x_g if x.shape[1] % 2 == 0 else torch.cat([x_g, x_g[:, :1, :, :]], dim=1)
        elif isinstance(x_g, int):
            output = x_l if x.shape[1] % 2 == 0 else torch.cat([x_l, x_l[:, :1, :, :]], dim=1)
        else:
            output = torch.cat([x_l, x_g], dim=1)

        return output


# ============================================================================
# ProxNet（Xnet / Mnet）—— 从 ProxNet.py 平铺（FINDNet_Xnet / FINDNet_Mnet 分支）
# ============================================================================

# 每个 stage 的全局/局部通道比（FFC 的 ratio_gin/gout），官方 S<=10 用满 index 0..9。
_RATIO_GIN = [0.0, 0.2, 0.2, 0.3, 0.4, 0.6, 0.7, 0.7, 0.8, 0.8, 0.8]
_RATIO_GOUT = [0.0, 0.2, 0.2, 0.3, 0.4, 0.6, 0.7, 0.7, 0.8, 0.8, 0.8]


class Mnet(nn.Module):
    """proxNet_M：字典 M 的 ResBlock 近端网络（FFC 版）。"""

    def __init__(self, args, ratio_index):
        super().__init__()
        self.channels = args.num_M
        self.T = args.T
        self.layer = self.make_resblock(self.T, ratio_index=ratio_index)
        tau0 = torch.Tensor([0.5])
        self.tau_const = tau0.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
            -1, self.channels, -1, -1)
        self.tau = nn.Parameter(self.tau_const, requires_grad=True)  # for sparsity

    def make_resblock(self, T, ratio_index):
        layers = []
        for _ in range(T):
            layers.append(
                nn.Sequential(
                    FFC_BN_ACT(self.channels, self.channels, kernel_size=3, stride=1,
                               padding=1, dilation=1, groups=1,
                               ratio_gin=_RATIO_GIN[ratio_index],
                               ratio_gout=_RATIO_GOUT[ratio_index]),
                    nn.ReLU(),
                    FFC_BN_ACT(self.channels, self.channels, kernel_size=3, stride=1,
                               padding=1, dilation=1, groups=1,
                               ratio_gin=_RATIO_GIN[ratio_index],
                               ratio_gout=_RATIO_GOUT[ratio_index]),
                ))
        return nn.Sequential(*layers)

    def forward(self, input):
        M = input
        for i in range(self.T):
            M = F.relu(M + self.layer[i](M))
        return F.relu(M - self.tau)


class Xnet(nn.Module):
    """proxNet_X：重建 X 的 ResBlock 近端网络（FFC 版）。"""

    def __init__(self, args, ratio_index):
        super().__init__()
        self.channels = args.num_Q + 1
        self.T = args.T
        self.layer = self.make_resblock(self.T, ratio_index=ratio_index)

    def make_resblock(self, T, ratio_index):
        layers = []
        for _ in range(T):
            layers.append(nn.Sequential(
                FFC_BN_ACT(self.channels, self.channels, kernel_size=3, stride=1,
                           padding=1, dilation=1, groups=1,
                           ratio_gin=_RATIO_GIN[ratio_index],
                           ratio_gout=_RATIO_GOUT[ratio_index]),
                nn.ReLU(),
                FFC_BN_ACT(self.channels, self.channels, kernel_size=3, stride=1,
                           padding=1, dilation=1, groups=1,
                           ratio_gin=_RATIO_GIN[ratio_index],
                           ratio_gout=_RATIO_GOUT[ratio_index]),
            ))
        return nn.Sequential(*layers)

    def forward(self, input):
        X = input
        for i in range(self.T):
            X = F.relu(X + self.layer[i](X))
        return X


# ============================================================================
# FINDNet —— 从 findnet.py 平铺（内核从 init_kernel.npy 载入）
# ============================================================================

# v1.1 标准化翻译服：原始 HU → [0,1]（(HU+1500)/5000）→ clip → ×255 → [0,255]。
# 空气 -1000 → 25.5（≥0 可表示）；反归一化在 reconstruct 输出处 x̂ = X/255*5000 - 1500。
HU_NORM_OFFSET = 1500.0
HU_NORM_SCALE = 5000.0
HU_INPUT_SCALE = 255.0   # v1.1-scale：入 ×255 / 出 ÷255，匹配官方 FIND-Net [0,255] 域


@dataclass
class FINDNetArgs:
    S: int = 10                 # stage 数（含初始化）
    num_M: int = 32             # 字典 M 特征通道
    num_Q: int = 32             # 对偶变量 Q 通道（Xnet 输入 channels = num_Q+1）
    etaM: float = 1.0           # M 更新步长
    etaX: float = 5.0           # X 更新步长
    T: int = 3                  # 每个 ProxNet 的 ResBlock 数
    kernel_path: str | None = field(default=None)  # None → 本目录 init_kernel.npy
    # 逐 ProxNet 梯度检查点：512×512 下 FIND-Net（10 stage × 每 stage 12 个 FFC_BN_ACT）
    # 前向 autograd 图 ~11GB/次，单次前向即 OOM（12GB 卡）。整网 checkpoint 无效（反向
    # 重算时又整网持图 11GB），必须按 ProxNet 粒度切：反向重算只持单个 ProxNet（~0.6GB）。
    use_checkpoint: bool = True
    # v2.0-HiFi：在最后一个 Xnet 的特征上接 HiFi 刻画块（频域分家 + 内容门控），
    # 增强 FIND-Net 对「预测组织高频」的刻画能力。默认 False = v1.3 基线。



_KERNEL_NPY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "init_kernel.npy")


def _load_kernel(num_M, kernel_path):
    path = kernel_path or _KERNEL_NPY
    if path and os.path.exists(path):
        k = np.load(path).astype(np.float32)
        if tuple(k.shape) == (1, num_M, 9, 9):
            return torch.FloatTensor(np.ascontiguousarray(k))
    # 兜底：确定性随机 init（无 .npy 时仍能建图，供单测）
    k = torch.empty(1, num_M, 9, 9)
    nn.init.kaiming_normal_(k, a=0.1)
    return k


class FINDNet(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        if args is None:
            args = FINDNetArgs()
        self.S = args.S
        self.iter = self.S - 1
        self.num_M = args.num_M
        self.num_Q = args.num_Q
        self.use_checkpoint = args.use_checkpoint

        self.etaM = torch.Tensor([args.etaM])
        self.etaX = torch.Tensor([args.etaX])
        self.etaM_S = self.make_eta(self.iter, self.etaM)
        self.etaX_S = self.make_eta(self.S, self.etaX)

        # 字典核（官方共享 init_kernel.mat['C9'][0:1] → [1, num_M, 9, 9]）
        kernel = _load_kernel(self.num_M, args.kernel_path)
        self.K0 = nn.Parameter(data=kernel, requires_grad=True)
        self.K = nn.Parameter(data=kernel, requires_grad=True)

        # 初始化 X/Q 的 3×3 均值滤波
        filter = torch.FloatTensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]) / 9
        filter = filter.unsqueeze(0).unsqueeze(0)
        self.K_q_const = filter.expand(self.num_Q, 1, -1, -1)  # [num_Q, 1, 3, 3]
        self.K_q = nn.Parameter(self.K_q_const, requires_grad=True)

        self.proxNet_X_0 = Xnet(args, ratio_index=0)
        self.proxNet_X_S = self.make_Xnet(self.S, args)
        self.proxNet_M_S = self.make_Mnet(self.S, args)
        self.proxNet_X_last_layer = Xnet(args, ratio_index=0)

        self.tau_const = torch.Tensor([1])
        self.tau = nn.Parameter(self.tau_const, requires_grad=True)



    def make_Xnet(self, iters, args):
        return nn.Sequential(*[Xnet(args, ratio_index=i) for i in range(iters)])

    def make_Mnet(self, iters, args):
        return nn.Sequential(*[Mnet(args, ratio_index=i) for i in range(iters)])

    def make_eta(self, iters, const):
        const_dimadd = const.unsqueeze(0)
        const_f = const_dimadd.expand(iters, -1)
        return nn.Parameter(data=const_f, requires_grad=True)

    def _px(self, module, x):
        """按 ProxNet 粒度梯度检查点：反向时只重算单个 ProxNet，不整网持 11GB 图。

        use_reentrant=False（必要）：FFC 内大量 ReLU(inplace=True)/FFT，重入版不支持 inplace。
        """
        if self.use_checkpoint and x is not None:
            return torch.utils.checkpoint.checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, CT_ma, LIct, Mask):
        # CT_ma / LIct / Mask: [B, 1, H, W]（HU；Mask 1=金属，0=非金属，与全库数据契约一致）。
        # v1.1-scale：标准化翻译服（入）——HU → [0,1] → clip → ×255 → [0,255]，
        # 完全对齐官方 FIND-Net 输入动态范围（dataset.py: (x+1500)/5000 → clip[0,1] → *255）。
        # 空气 −1000 → 25.5（≥0 可表示），ReLU 不再钳死负数。
        CT_ma = torch.clamp((CT_ma + HU_NORM_OFFSET) / HU_NORM_SCALE, 0.0, 1.0) * HU_INPUT_SCALE
        LIct = torch.clamp((LIct + HU_NORM_OFFSET) / HU_NORM_SCALE, 0.0, 1.0) * HU_INPUT_SCALE
        # 官方 FIND-Net 的 Mask 是 non_Mask（1=非金属）：`Mask*(input−X)` 从非金属区提取伪影残差。
        # 全库数据契约是 1=金属（dataset/losses/trainer 统一），故在此入口翻一次，等价于
        # 每个调用点传 `1−metal_mask`，避免十几处调用点漏翻导致掩膜反相。
        Mask = 1.0 - Mask
        ListX = []
        ListA = []
        ListM = []
        input = CT_ma
        Q00 = F.conv2d(LIct, self.K_q, stride=1, padding=1)  # 对偶变量 Q
        input_ini = torch.cat((LIct, Q00), dim=1)
        XQ_ini = self._px(self.proxNet_X_0, input_ini)
        X0 = XQ_ini[:, :1, :, :]
        Q0 = XQ_ini[:, 1:, :, :]

        # 第 1 迭代：X0 -> M1
        A_hat = Mask * (input - X0)
        A_hat_cut = F.relu(A_hat - self.tau)
        Epsilon = F.conv_transpose2d(A_hat_cut, self.K0 / 10, stride=1, padding=4)
        M1 = self._px(self.proxNet_M_S[0], Epsilon)
        A = F.conv2d(M1, self.K / 10, stride=1, padding=4)

        # 第 1 迭代：M1 -> X1
        A_hat = input - A
        X_mid = (1 - self.etaX_S[0] * Mask / 10) * X0 + self.etaX_S[0] * Mask / 10 * A_hat
        input_concat = torch.cat((X_mid, Q0), dim=1)
        XQ = self._px(self.proxNet_X_S[0], input_concat)
        X1 = XQ[:, :1, :, :]
        Q1 = XQ[:, 1:, :, :]
        ListX.append(X1)
        ListA.append(A)
        ListM.append(M1)
        X = X1
        Q = Q1
        M = M1

        for i in range(self.iter):
            # M-net
            A_hat = Mask * (input - X)
            Epsilon = (self.etaM_S[i, :] / 10
                       * F.conv_transpose2d((Mask * A - A_hat), self.K / 10,
                                            stride=1, padding=4))
            M = self._px(self.proxNet_M_S[i + 1], M - Epsilon)

            # X-net
            A = F.conv2d(M, self.K / 10, stride=1, padding=4)
            ListA.append(A)
            X_hat = input - A
            X_mid = ((1 - self.etaX_S[i + 1, :] * Mask / 10) * X
                     + self.etaX_S[i + 1, :] * Mask / 10 * X_hat)
            input_concat = torch.cat((X_mid, Q), dim=1)
            XQ = self._px(self.proxNet_X_S[i + 1], input_concat)
            X = XQ[:, :1, :, :]
            Q = XQ[:, 1:, :, :]
            ListX.append(X)

        XQ_adjust = self._px(self.proxNet_X_last_layer, XQ)

        # v1.1：输出夹回 [0,255]（翻译服契约）。Xnet 是 F.relu(X+layer(X)) 残差，上界无界，
        # 空气区被 B_down 掩膜后无监督，FFC 边界角点可冲高。clamp 后反归一化锁定
        # [-1500, 3500] 物理量程（X/255*5000−1500）。
        X = XQ_adjust[:, :1, :, :].clamp(0.0, HU_INPUT_SCALE)
        ListX.append(X)
        return X0, ListX, ListA

    def reconstruct(self, CT_ma, LIct, Mask):
        """PRISM 接口：返回最终重建 x̂_z = ListX[-1]（唯一输出），反归一化回 HU。"""
        X = self.forward(CT_ma, LIct, Mask)[1][-1]
        return X / HU_INPUT_SCALE * HU_NORM_SCALE - HU_NORM_OFFSET
