#!/usr/bin/env python
"""eval_mar.py —— held-out 病人的 MAR 分区评测（C 消融 H2 的判据，也供 A–F 共用）。

加载 run_train.py 产出的 checkpoint（MNetwork + FIND-Net + config 里的冻结 d_ref），对
指定 held-out 病人的每个植入体序列跑双向递归（FIND-Net 在循环内当观测），得到唯一
最终输出 x̂_z，再与非金属真值 X_clean 按数据充分度 d_j/d_ref 三分区报告 RMSE / PSNR：

    数据充分区  d_j/d_ref > 9
    部分阴影区  d_j/d_ref ∈ [1/9, 9]   ← H2 的核心判据区
    完全阴影区  d_j/d_ref < 1/9

重要：d_ref 必须来自「训练集（排除 held-out 病人）冻结」的 checkpoint —— 即训练时用
`--holdout-patient <patient>` 跑出的模型。否则 d_ref 会混入 held-out 病人的统计（泄漏），
分区边界与训练态不一致。

用法：
    python scripts/eval_mar.py \
        --checkpoint /root/autodl-tmp/PRISM/training_data/ssm_nb1.pt \
        --data-dir /root/autodl-tmp/PRISM/training_data \
        --patient patients_head2 --trust-mode kalman \
        --nb 1 --channels 32 --device cuda
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_FRAMEWORK_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _FRAMEWORK_ROOT)

from prism.data.dataset import make_theta_sequence_loader  # noqa: E402
from prism.models.m_network import MNetwork  # noqa: E402
from prism.models.find_net import FINDNet, FINDNetArgs  # noqa: E402
from prism.recursion.algorithm1 import forward_backward  # noqa: E402
from prism.training.trainer import SSMTrainer  # noqa: E402
from prism.utils.config import SSMConfig  # noqa: E402


def _sample_idx(sample_id: str) -> int:
    return int(sample_id.rsplit("_", 1)[1])


def transition_zone_mask(metal_mask, r, dilate_px=10, r_low=1.0 / 9.0):
    """过渡区掩膜 = 金属外扩环带 ∩ 光子饥饿区（靶向评价体系第三部分）。

    物理定义：紧贴金属外缘、光子饥饿（r=d_j/d_ref < r_low）最严重的一圈——2D 信息近乎 0、
    条纹侵蚀最重、医生最关注（瘤颈/载瘤动脉）。

    实现（纯 torch，无 scipy/torchvision）：metal_mask 二值化 → 3D max-pool 外扩 dilate_px
    体素 → 减原金属核得环带 → 与低 d_j（r<r_low）取交。
    metal_mask/r: [B,1,Z,H,W]；返回同形 [B,1,Z,H,W]（0/1）。
    """
    binm = (metal_mask > 0.5).float()
    k = 2 * dilate_px + 1
    dilated = F.max_pool3d(binm, kernel_size=(1, k, k), stride=1,
                           padding=(0, dilate_px, dilate_px))
    ring = (dilated - binm).clamp(min=0.0)
    low = (r < r_low).float()
    return ring * low


def build_patient_specs(meta_dir: str, patient_id: str, min_z: int = 3, max_z: int = 0):
    """只取指定病人（include）的样本，按 sequence_id 分组 → specs。"""
    metas = []
    for f in glob.glob(os.path.join(meta_dir, "*.json")):
        m = json.load(open(f))
        if m.get("patient_id") == patient_id:
            metas.append(m)
    by_seq = {}
    for m in metas:
        by_seq.setdefault(m["sequence_id"], []).append(m["sample_id"])
    specs = [sorted(sids, key=_sample_idx) for sids in by_seq.values()]
    specs = [s for s in specs if len(s) >= min_z]
    if max_z > 0:
        specs = [s[:max_z] for s in specs]
    specs.sort(key=lambda s: (_sample_idx(s[0]), len(s)))
    return specs, metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data")
    ap.add_argument("--baseline-dir", default=None,
                    help="覆盖观测目录（默认 train/Baseline）；OOD 检查指向 Baseline_bcorr 等")
    ap.add_argument("--patient", required=True, help="held-out 病人 patient_id")
    ap.add_argument("--trust-mode", default="kalman", choices=["kalman", "gated", "fixed"])
    ap.add_argument("--recursion", default="bidirectional",
                    choices=["bidirectional", "forward", "per_slice"],
                    help="B 消融（H3）递归模式，须与 checkpoint 训练时一致")
    ap.add_argument("--nb", type=int, default=1)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min-z", type=int, default=3)
    ap.add_argument("--max-z", type=int, default=0)
    ap.add_argument("--max-implants", type=int, default=0,
                    help=">0 时只评前 N 植入体（smoke）")
    ap.add_argument("--out-json", default=None, help="可选：把分区指标写到 JSON")
    ap.add_argument("--gain-mode", default="kalman", choices=["kalman", "gated", "fixed"],
                    help="A3 增益替换（默认取 checkpoint 存的；CLI 覆盖）")
    ap.add_argument("--findnet-lict", default="mu", choices=["mu", "li"],
                    help="A2 去先验（默认取 checkpoint 存的；CLI 覆盖）")
    ap.add_argument("--transition-dilate-px", type=int, default=10,
                    help="过渡区环带外扩像素数（金属核外扩减原核）")
    ap.add_argument("--transition-r-low", type=float, default=1.0 / 9.0,
                    help="过渡区光子饥饿阈值 r=d_j/d_ref < r_low")
    args = ap.parse_args()

    # 1. 载入 checkpoint
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = ckpt["config"]
    d_ref = cfg["d_ref"]
    nb = cfg.get("nb", args.nb)
    channels = cfg.get("channels", args.channels)
    use_velocity = cfg.get("use_velocity", True)
    gate_reads_dj = cfg.get("gate_reads_dj", False)
    ablate_xm0 = cfg.get("ablate_xm0", False)
    ablate_dj = cfg.get("ablate_dj", False)
    ablate_g = cfg.get("ablate_g", False)
    gain_mode = cfg.get("gain_mode", args.gain_mode)
    findnet_lict_source = cfg.get("findnet_lict_source", args.findnet_lict)
    binarize_mask = cfg.get("binarize_mask", False)
    mask_bin_thr = cfg.get("mask_bin_thr", 0.5)
    recursion = cfg.get("recursion", args.recursion)

    m_net = MNetwork(nb=nb, channels=channels, use_velocity=use_velocity,
                     gate_reads_dj=gate_reads_dj,
                     ablate_xm0=ablate_xm0, ablate_dj=ablate_dj, ablate_g=ablate_g).to(args.device)
    find_net = FINDNet(FINDNetArgs()).to(args.device)
    m_net.load_state_dict(ckpt["model"])
    find_net.load_state_dict(ckpt["find_net"])
    m_net.eval()
    find_net.eval()

    # 复现训练态：递归循环定案后 FIND-Net 在循环内当观测，不显式吃 trust 通道；
    # trust_mode 仅保留旧 C 消融诊断脚本兼容（checkpoint 不存 trust_mode，CLI 指定）。
    config = SSMConfig(nb=nb, channels=channels, trust_mode=args.trust_mode,
                       recursion=recursion, d_ref=d_ref,
                       gain_mode=gain_mode, findnet_lict_source=findnet_lict_source,
                       binarize_mask=binarize_mask, mask_bin_thr=mask_bin_thr)
    trainer = SSMTrainer(m_net, config, device=args.device, find_net=find_net)

    print(f"[eval] checkpoint={args.checkpoint}\n"
          f"  patient={args.patient!r}  trust_mode={args.trust_mode}  nb={nb}  "
          f"d_ref={d_ref:.6g}  gain_mode={gain_mode}  findnet_lict={findnet_lict_source}  "
          f"recursion={recursion}  binarize={binarize_mask}", flush=True)

    # 2. 组 held-out 病人序列
    train_dir = os.path.join(args.data_dir, "train")
    meta_dir = os.path.join(args.data_dir, "meta")
    specs, metas = build_patient_specs(meta_dir, args.patient,
                                        min_z=args.min_z, max_z=args.max_z)
    if args.max_implants > 0:
        specs = specs[: args.max_implants]
    print(f"[eval] {args.patient!r}: {len(specs)} 序列 / {len(metas)} 样本", flush=True)

    baseline_dir = os.path.join(train_dir, "Baseline") if not args.baseline_dir else args.baseline_dir
    loader = make_theta_sequence_loader(
        os.path.join(train_dir, "Target"), baseline_dir,
        os.path.join(train_dir, "LI"), os.path.join(train_dir, "Mask"),
        os.path.join(train_dir, "Phys"))

    # 三分区累计（非金属区）：充分 / 部分阴影 / 完全阴影；另加过渡区与全局。
    SSE = {"充分": 0.0, "部分阴影": 0.0, "完全阴影": 0.0}
    CNT = {"充分": 0, "部分阴影": 0, "完全阴影": 0}
    tz_sse = 0.0       # 过渡区 SSE
    tz_sumerr = 0.0    # 过渡区 signed error 和（Bias）
    tz_cnt = 0
    glob_sse = 0.0     # 全局非金属 SSE
    glob_cnt = 0
    peak = 0.0

    with torch.no_grad():
        for i, spec in enumerate(specs):
            sample = loader(spec)
            thetas = [{k: v.unsqueeze(0).to(args.device) for k, v in t.items()}
                      for t in sample["thetas"]]           # [Z] dict, [1,1,H,W]
            x_hat = sample["x_hat"].unsqueeze(0).to(args.device)      # [1,Z,1,H,W]
            # X_clean / metal_mask 统一到 [1,1,Z,H,W]，与 x_hat_final [1,1,Z,H,W] 对齐。
            # 直接 [1,Z,1,H,W] 会让 (x_hat_final − X_clean) 广播成 [1,Z,Z,H,W]（每片对每片），
            # 体素数膨胀 Z 倍、RMSE 全错。
            X_clean = sample["X_clean"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(args.device)
            metal_mask = sample["metal_mask"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(args.device)
            li = sample["li"].unsqueeze(0).to(args.device)             # [1,Z,1,H,W]

            Z = len(thetas)
            dirty_vol = x_hat.permute(0, 2, 1, 3, 4)                   # [1,1,Z,H,W]
            dirty = [dirty_vol[:, :, z] for z in range(Z)]             # [Z] 个 [1,1,H,W]
            li_vol = li.permute(0, 2, 1, 3, 4)                         # [1,1,Z,H,W]
            li_list = [li_vol[:, :, z] for z in range(Z)]
            # B1：eval 复现训练态——findnet_lict_source="li"（A2）用 LI；binarize 喂 FIND-Net 的 Mask。
            mask_input = (metal_mask if not config.binarize_mask
                          else (metal_mask > config.mask_bin_thr).float())
            masks = [mask_input[:, :, z] for z in range(Z)]            # [Z] 个 [1,1,H,W]
            observer = trainer._observer(dirty, masks, li_list)
            res = forward_backward(m_net, thetas, d_ref,
                                   recursion=recursion, observer=observer,
                                   gain_mode=config.gain_mode)
            x_hat_final = trainer._final_output(res, dirty, masks, li_list)  # [1,1,Z,H,W]

            I = 1.0 - metal_mask                                       # 非金属=1
            d_j = trainer._stack([t["d_j"] for t in thetas])           # [1,1,Z,H,W]
            r = d_j / d_ref
            err = (x_hat_final - X_clean) ** 2                         # [1,1,Z,H,W]
            signed_err = x_hat_final - X_clean                         # [1,1,Z,H,W]
            peak = max(peak, float(X_clean[I > 0.5].max().item()))

            for name in SSE:
                sel = I > 0.5
                sel = sel & (r > 9.0) if name == "充分" else sel
                sel = sel & (r < 1.0 / 9.0) if name == "完全阴影" else sel
                if name == "部分阴影":
                    sel = sel & (r >= 1.0 / 9.0) & (r <= 9.0)
                SSE[name] += float(err[sel].sum().item())
                CNT[name] += int(sel.sum().item())

            # 过渡区（金属外扩环带 ∩ 光子饥饿）与全局非金属。
            tz = transition_zone_mask(metal_mask, r, args.transition_dilate_px,
                                      args.transition_r_low)
            tz_sse += float((err * tz).sum().item())
            tz_sumerr += float((signed_err * tz).sum().item())
            tz_cnt += int(tz.sum().item())
            g = I > 0.5
            glob_sse += float(err[g].sum().item())
            glob_cnt += int(g.sum().item())

            if (i + 1) % 25 == 0:
                print(f"  ... {i + 1}/{len(specs)} 序列", flush=True)

    tz_rmse = float(np.sqrt(tz_sse / tz_cnt)) if tz_cnt > 0 else float("nan")
    tz_bias = float(tz_sumerr / tz_cnt) if tz_cnt > 0 else float("nan")
    glob_rmse = float(np.sqrt(glob_sse / glob_cnt)) if glob_cnt > 0 else float("nan")

    print(f"\n[结果] peak(HU)={peak:.1f}", flush=True)
    print(f"{'分区':<8} {'体素数':>10} {'RMSE(HU)':>12} {'Bias(HU)':>10} {'PSNR(dB)':>10}",
          flush=True)
    tz_psnr = float(20.0 * np.log10(peak / tz_rmse)) if (tz_cnt > 0 and tz_rmse > 0) else float("nan")
    print(f"{'过渡区':<8} {tz_cnt:>10} {tz_rmse:>12.3f} {tz_bias:>10.1f} {tz_psnr:>10.2f}",
          flush=True)
    results = {"过渡区": {"n_voxels": tz_cnt, "rmse": tz_rmse, "bias": tz_bias,
                          "psnr": tz_psnr}}
    for name in SSE:
        cnt = CNT[name]
        rmse = float(np.sqrt(SSE[name] / cnt)) if cnt > 0 else float("nan")
        psnr = float(20.0 * np.log10(peak / rmse)) if (cnt > 0 and rmse > 0) else float("nan")
        results[name] = {"n_voxels": cnt, "rmse": rmse, "psnr": psnr}
        print(f"{name:<8} {cnt:>10} {rmse:>12.3f} {'-':>10} {psnr:>10.2f}", flush=True)
    gpsnr = float(20.0 * np.log10(peak / glob_rmse)) if (glob_cnt > 0 and glob_rmse > 0) else float("nan")
    results["全局"] = {"n_voxels": glob_cnt, "rmse": glob_rmse, "psnr": gpsnr}
    print(f"{'全局':<8} {glob_cnt:>10} {glob_rmse:>12.3f} {'-':>10} {gpsnr:>10.2f}", flush=True)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"patient": args.patient, "trust_mode": args.trust_mode,
                       "gain_mode": config.gain_mode,
                       "findnet_lict": config.findnet_lict_source,
                       "d_ref": d_ref, "peak_hu": peak, "results": results,
                       "n_implants": len(specs)}, f, indent=2)
        print(f"[save] {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
