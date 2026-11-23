"""v22_metrics_table.py — 四尺寸 × 三个状态（植入前/植入后/FIND-Net）的 MAE/SSIM/PSNR。

- 植入前 = 零金属重建（同一管线，无金属）：与 GT 的差 = 重建/噪声地板（不是伪影损失）
- 植入后 = Baseline（带伪影）
- FIND-Net = v2.1 + 金属回填（管线真实输出）

指标口径：
  MAE / RMSE / PSNR 在**非金属组织**上算（剔除 metal + 空气），data_range=3000
  SSIM 在**整幅 512×512** 上算（data_range=3000）——SSIM 需要完整窗口
  信息缺失度 = 组织内 |err| > 50 HU 的体素占比
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from prism.eval.domain import tissue_mask
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

sys.path.insert(0, "/root/autodl-tmp/prism_framework")
from prism.models.find_net import FINDNet, FINDNetArgs

D = "/root/autodl-tmp/PRISM/training_data_astra"
CACHE = "/root/autodl-tmp/PRISM/v2.2/info_loss/_before_cache"
OUT = "/root/autodl-tmp/PRISM/v2.2/info_loss"
PICKS = {"small": "sample_003090", "medium": "sample_003169",
         "large": "sample_003260", "giant": "sample_001308"}
DR = 3000.0


def rd(sid, sub):
    p = os.path.join(D, "train", sub, f"{sid}.raw")
    if not os.path.exists(p):
        p = os.path.join(D, "val", sub, f"{sid}.raw")
    return np.fromfile(p, np.float32).reshape(512, 512)


def backfill(pred, baseline, mask, r=8):
    b = torch.from_numpy((mask > 0.5).astype(np.float32))[None, None]
    dil = F.max_pool2d(b, kernel_size=2 * r + 1, stride=1, padding=r)[0, 0].numpy()
    return (1.0 - dil) * pred + dil * baseline


def ssim_np(x, y, dr=3000.0, sigma=1.5):
    """标准 SSIM（高斯窗 σ=1.5, C1=(0.01·dr)², C2=(0.03·dr)²）。"""
    x = x.astype(np.float64); y = y.astype(np.float64)
    C1 = (0.01 * dr) ** 2; C2 = (0.03 * dr) ** 2
    mx = gaussian_filter(x, sigma); my = gaussian_filter(y, sigma)
    sxx = gaussian_filter(x * x, sigma) - mx ** 2
    syy = gaussian_filter(y * y, sigma) - my ** 2
    sxy = gaussian_filter(x * y, sigma) - mx * my
    s = ((2 * mx * my + C1) * (2 * sxy + C2)) /         ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))
    return float(s.mean())


def metrics(pred, gt, sel):
    e = pred - gt
    mae = float(np.abs(e)[sel].mean())
    rmse = float(np.sqrt((e[sel] ** 2).mean()))
    psnr = float(20 * np.log10(DR / max(rmse, 1e-6)))
    ssim = ssim_np(gt, pred, DR)
    frac50 = float((np.abs(e)[sel] > 50).mean())
    return mae, rmse, psnr, ssim, frac50


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = FINDNet(FINDNetArgs()).to(dev)
    net.load_state_dict(torch.load(f"{D}/findnet_stage1_v2.1.pt", map_location=dev,
                                   weights_only=False)["find_net"], assign=True)
    net.eval()
    rows = []
    print(f"{'size':7s} {'state':12s} {'MAE':>7s} {'RMSE':>7s} {'PSNR':>6s} {'SSIM':>6s} {'>50HU':>6s}")
    for size, sid in PICKS.items():
        gt, bl, li, mk = (rd(sid, s) for s in ("Target", "Baseline", "LI", "Mask"))
        cf_p = os.path.join(CACHE, f"{sid}_cleanFBP.npy")
        cf = np.load(cf_p).astype(np.float32) if os.path.exists(cf_p) else None
        t = lambda a: torch.from_numpy(a[None, None].astype(np.float32)).to(dev)
        with torch.no_grad():
            raw = net.reconstruct(t(bl), t(li), t(mk))[0, 0].cpu().numpy()
        fn = backfill(raw, bl, mk)
        sel = tissue_mask(mk, gt, r=8)
        states = [("FIND-Net", fn, bl)] if False else []
        seq = []
        if cf is not None:
            seq.append(("before(clean FBP)", cf))
        seq.append(("after(Baseline)", bl))
        seq.append(("FIND-Net", fn))
        for name, img in seq:
            mae, rmse, psnr, ssim, f50 = metrics(img, gt, sel)
            rows.append((size, name, mae, rmse, psnr, ssim, f50))
            print(f"{size:7s} {name:12s} {mae:7.1f} {rmse:7.1f} {psnr:6.2f} {ssim:6.3f} {f50:6.3f}")
        print()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "metrics_mae_ssim_psnr.csv"), "w") as f:
        f.write("size,state,mae,rmse,psnr,ssim,frac_gt50\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.3f},{r[3]:.3f},{r[4]:.3f},{r[5]:.5f},{r[6]:.5f}\n")
    print("[save]", os.path.join(OUT, "metrics_mae_ssim_psnr.csv"))


if __name__ == "__main__":
    main()
