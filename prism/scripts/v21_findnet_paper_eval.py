#!/usr/bin/env python
"""v21_findnet_paper_eval.py —— 仿 FIND-Net 原论文 Table 1 的指标评测。

对比：
  (1) 原论文预训练权重 FINDNet_noGF_orig.pt（BatchNorm，No-GF 分支，同架构）；
  (2) 我们刚训的 findnet_stage1_v2.1.pt（GroupNorm，完全收敛）。
在 held-out（patients_head2，训练时 --holdout-patient 排除）上，按金属尺寸(category)
分层算 MAE↓ / SSIM↑ / PSNR↑，data_range=3000，masked_eval（金属区置 0，对齐原论文）。

注意：原权重是 BN 架构，我们的 port 是 GN——两者架构同（只是归一化层不同），
这里都跑在各自原生的架构上（原权重用 find_net_bn.py，我们用 find_net.py），公平。
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/prism_framework")

from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr  # noqa: E402

D = "/root/autodl-tmp/PRISM/training_data_astra/train"
META = "/root/autodl-tmp/PRISM/training_data_astra/meta"
ORIG = "/root/autodl-tmp/prism_framework/FINDNet_noGF_orig.pt"
OURS = "/root/autodl-tmp/PRISM/training_data_astra/findnet_stage1_v2.1.pt"
DATA_RANGE = 3000.0


def rd(sid, sub):
    return np.fromfile(os.path.join(D, sub, f"{sid}.raw"), np.float32).reshape(512, 512)


def hu255(x):
    return np.clip((x + 1500.0) / 5000.0, 0.0, 1.0) * 255.0


def tohu(x):
    return x / 255.0 * 5000.0 - 1500.0


def calc(gt, pred, mask_nm):
    gtm = gt * mask_nm
    prm = pred * mask_nm
    mae = float(np.mean(np.abs(gtm - prm)))
    s = float(ssim(gtm, prm, data_range=DATA_RANGE))
    p = float(psnr(gtm, prm, data_range=DATA_RANGE))
    return mae, s, p


def main():
    from prism.models.find_net_bn import FINDNet as FN_BN, FINDNetArgs as A_BN  # noqa: E402
    from prism.models.find_net import FINDNet as FN_GN, FINDNetArgs as A_GN  # noqa: E402

    dev = "cuda"
    # 原论文权重（BN，No-GF），use_checkpoint=False 加速推理
    m_orig = FN_BN(A_BN(use_checkpoint=False)).to(dev)
    sd = torch.load(ORIG, map_location=dev, weights_only=False)
    if isinstance(sd, dict) and "find_net" in sd:
        sd = sd["find_net"]
    m_orig.load_state_dict(sd, strict=True, assign=True)
    m_orig.eval()

    # 我们 v2.1（GN）
    m_ours = FN_GN(A_GN(use_checkpoint=False)).to(dev)
    sd2 = torch.load(OURS, map_location=dev, weights_only=False)
    if isinstance(sd2, dict) and "find_net" in sd2:
        sd2 = sd2["find_net"]
    m_ours.load_state_dict(sd2, strict=True)
    m_ours.eval()
    print("[load] 原论文权重 + 我们 v2.1 权重 OK", flush=True)

    # held-out = patients_head2
    metas = [json.load(open(f)) for f in glob.glob(os.path.join(META, "*.json"))]
    ho = [m for m in metas if m["patient_id"] == "patients_head2"]
    print(f"[eval] held-out patients_head2 样本数 = {len(ho)}", flush=True)

    agg = defaultdict(lambda: [[], []])   # cat -> [orig_list, ours_list]

    for i, meta in enumerate(ho):
        sid = meta["sample_id"]
        cat = meta["category"]
        base = rd(sid, "Baseline")
        tgt = rd(sid, "Target")
        li = rd(sid, "LI")
        msk = rd(sid, "Mask")
        mask_nm = (msk < 0.5).astype(np.float32)      # 1=非金属（masked_eval 用）

        # 原论文权重（BN）：输入 [0,255]，Mask=非金属(1=非金属)
        with torch.no_grad():
            b_t = torch.from_numpy(hu255(base)[None, None]).to(dev)
            l_t = torch.from_numpy(hu255(li)[None, None]).to(dev)
            m_t = torch.from_numpy(mask_nm[None, None]).to(dev)
            _, ListX, _ = m_orig(b_t, l_t, m_t)
            pred_orig = tohu(ListX[-1][0, 0].cpu().numpy())

        # 我们 v2.1（GN）：输入 HU，Mask=金属(1=金属)，reconstruct 内部归一化
        with torch.no_grad():
            b_t = torch.from_numpy(base[None, None]).to(dev)
            l_t = torch.from_numpy(li[None, None]).to(dev)
            m_t = torch.from_numpy((msk > 0.5).astype(np.float32)[None, None]).to(dev)
            pred_ours = m_ours.reconstruct(b_t, l_t, m_t)[0, 0].cpu().numpy()

        mo = calc(tgt, pred_orig, mask_nm)
        mu = calc(tgt, pred_ours, mask_nm)
        agg[cat][0].append(mo)
        agg[cat][1].append(mu)
        if (i + 1) % 60 == 0:
            print(f"  ...{i + 1}/{len(ho)}", flush=True)

    # 汇总表
    order = ["giant", "large", "medium", "small"]
    print("\n=== MAE↓/SSIM↑/PSNR↑（masked，data_range=3000，held-out patients_head2）===", flush=True)
    print(f"{'金属尺寸':8s} {'样本':>5s} | {'原论文(No-GF)':>26s} | {'我们 v2.1':>26s}", flush=True)
    print(f"{'':8s} {'':>5s} | {'MAE':>6s} {'SSIM':>6s} {'PSNR':>6s} {'':>5s} | "
          f"{'MAE':>6s} {'SSIM':>6s} {'PSNR':>6s}", flush=True)
    all_o, all_u = [], []
    for cat in order:
        if cat not in agg:
            continue
        o, u = agg[cat]
        mo = tuple(np.mean(np.array(o), axis=0))
        mu = tuple(np.mean(np.array(u), axis=0))
        all_o.extend(o)
        all_u.extend(u)
        print(f"{cat:8s} {len(o):5d} | {mo[0]:6.2f} {mo[1]:6.3f} {mo[2]:6.2f}      | "
              f"{mu[0]:6.2f} {mu[1]:6.3f} {mu[2]:6.2f}", flush=True)
    if all_o:
        mo = tuple(np.mean(np.array(all_o), axis=0))
        mu = tuple(np.mean(np.array(all_u), axis=0))
        print(f"{'平均':8s} {len(all_o):5d} | {mo[0]:6.2f} {mo[1]:6.3f} {mo[2]:6.2f}      | "
              f"{mu[0]:6.2f} {mu[1]:6.3f} {mu[2]:6.2f}", flush=True)


if __name__ == "__main__":
    main()
