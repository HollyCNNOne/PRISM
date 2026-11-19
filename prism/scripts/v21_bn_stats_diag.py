#!/usr/bin/env python
"""v21_bn_stats_diag.py —— 诊断：原 FIND-Net 权重崩溃的主因是 BN running-stats 失配还是 conv 权重本身。

方法：同一份原论文 BN 权重，对同一样本跑两种模式：
  eval()：BN 用 running-stats（他们 DeepLesion 数据的统计）→ 预期崩（SSIM≈0.19）。
  train()：BN 用 batch 统计（batch=1 → 近似 instance norm，逐样本归一化）→ 若明显更好，
           实锤崩溃主因是 running-stats 失配，而不是 conv 权重学坏了。
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/prism_framework")

from skimage.metrics import structural_similarity as ssim  # noqa: E402

D = "/root/autodl-tmp/PRISM/training_data_astra/train"
META = "/root/autodl-tmp/PRISM/training_data_astra/meta"
ORIG = "/root/autodl-tmp/prism_framework/FINDNet_noGF_orig.pt"


def rd(sid, sub):
    return np.fromfile(os.path.join(D, sub, f"{sid}.raw"), np.float32).reshape(512, 512)


def hu255(x):
    return np.clip((x + 1500.0) / 5000.0, 0.0, 1.0) * 255.0


def tohu(x):
    return x / 255.0 * 5000.0 - 1500.0


def ssim_mask(gt, pred, mask_nm):
    return float(ssim(gt * mask_nm, pred * mask_nm, data_range=3000.0))


def main():
    from prism.models.find_net_bn import FINDNet, FINDNetArgs  # noqa: E402
    dev = "cuda"
    m = FINDNet(FINDNetArgs(use_checkpoint=False)).to(dev)
    sd = torch.load(ORIG, map_location=dev, weights_only=False)
    if isinstance(sd, dict) and "find_net" in sd:
        sd = sd["find_net"]
    m.load_state_dict(sd, strict=True, assign=True)

    metas = [json.load(open(f)) for f in glob.glob(os.path.join(META, "*.json"))]
    ho = [mm for mm in metas if mm["patient_id"] == "patients_head2"]
    bycat = defaultdict(list)
    for mm in ho:
        bycat[mm["category"]].append(mm)
    sids = []
    for cat in ["giant", "large", "medium", "small"]:
        sids += bycat[cat][:3]
    print(f"[diag] {len(sids)} 个样本，每个 eval vs train 各跑一次", flush=True)

    ev, tr = [], []
    for meta in sids:
        sid = meta["sample_id"]
        base = rd(sid, "Baseline")
        tgt = rd(sid, "Target")
        li = rd(sid, "LI")
        msk = rd(sid, "Mask")
        mask_nm = (msk < 0.5).astype(np.float32)
        b = torch.from_numpy(hu255(base)[None, None]).to(dev)
        l = torch.from_numpy(hu255(li)[None, None]).to(dev)
        mn = torch.from_numpy(mask_nm[None, None]).to(dev)

        m.eval()
        with torch.no_grad():
            _, Lx, _ = m(b, l, mn)
            pe = tohu(Lx[-1][0, 0].cpu().numpy())
        m.train()
        with torch.no_grad():
            _, Lx, _ = m(b, l, mn)
            pt = tohu(Lx[-1][0, 0].cpu().numpy())

        se = ssim_mask(tgt, pe, mask_nm)
        st = ssim_mask(tgt, pt, mask_nm)
        ev.append(se)
        tr.append(st)
        print(f"  {sid} {meta['category']:6s}: eval(BN running-stats) SSIM={se:.3f}  "
              f"train(BN batch-stats) SSIM={st:.3f}", flush=True)

    print(f"\n[结论] eval 平均 SSIM={np.mean(ev):.3f}  train 平均 SSIM={np.mean(tr):.3f}", flush=True)


if __name__ == "__main__":
    main()
