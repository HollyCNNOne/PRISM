#!/usr/bin/env python
"""v21_findnet_paper_eval_4way.py —— 四权重 FIND-Net 论文式评测（2×2：No-GF/GF × 原/我们）。

四权重：
  原 No-GF  = FINDNet_noGF_orig.pt（BN，find_net_bn.py）
  我们 No-GF = findnet_stage1_v2.1.pt（GN，find_net.py）
  原 GF     = FINDNet_orig.pt（BN+GF，find_net_bn_gf.py）
  我们 GF   = findnet_stage1_v2.2_gf.pt（GN+GF，find_net_gf.py；训练中，缺文件则跳过）

在 held-out patients_head2 上，按金属尺寸(category) 分层算 MAE↓/SSIM↑/PSNR↑
（masked_eval，data_range=3000，对齐原论文）。缺文件自动跳过，便于训练前后复用。
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
DATA_RANGE = 3000.0

W = {
    "原 No-GF":  ("/root/autodl-tmp/prism_framework/FINDNet_noGF_orig.pt", "bn_nogf"),
    "我们 No-GF": ("/root/autodl-tmp/PRISM/training_data_astra/findnet_stage1_v2.1.pt", "gn_nogf"),
    "原 GF":     ("/root/autodl-tmp/prism_framework/FINDNet_orig.pt", "bn_gf"),
    "我们 GF":   ("/root/autodl-tmp/PRISM/training_data_astra/findnet_stage1_v2.2_gf.pt", "gn_gf"),
}


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


def load_model(mode, dev):
    if mode == "bn_nogf":
        from prism.models.find_net_bn import FINDNet, FINDNetArgs
        m = FINDNet(FINDNetArgs(use_checkpoint=False)).to(dev)
        load = lambda sd: m.load_state_dict(sd, strict=True, assign=True)
        fwd_mode = "bn"
    elif mode == "gn_nogf":
        from prism.models.find_net import FINDNet, FINDNetArgs
        m = FINDNet(FINDNetArgs(use_checkpoint=False)).to(dev)
        load = lambda sd: m.load_state_dict(sd, strict=True)
        fwd_mode = "gn"
    elif mode == "bn_gf":
        from prism.models.find_net_bn_gf import FINDNet, FINDNetArgs
        m = FINDNet(FINDNetArgs(use_checkpoint=False)).to(dev)
        load = lambda sd: m.load_state_dict(sd, strict=True, assign=True)
        fwd_mode = "bn"
    elif mode == "gn_gf":
        from prism.models.find_net_gf import FINDNet, FINDNetArgs
        m = FINDNet(FINDNetArgs(use_checkpoint=False)).to(dev)
        load = lambda sd: m.load_state_dict(sd, strict=True)
        fwd_mode = "gn"
    else:
        raise ValueError(mode)
    m.eval()
    return m, load, fwd_mode


def run(m, fwd_mode, base, li, mask_metal, dev):
    if fwd_mode == "bn":
        b = torch.from_numpy(hu255(base)[None, None]).to(dev)
        l = torch.from_numpy(hu255(li)[None, None]).to(dev)
        nm = torch.from_numpy((mask_metal < 0.5).astype(np.float32)[None, None]).to(dev)
        with torch.no_grad():
            _, Lx, _ = m(b, l, nm)
        return tohu(Lx[-1][0, 0].cpu().numpy())
    else:
        b = torch.from_numpy(base[None, None]).to(dev)
        l = torch.from_numpy(li[None, None]).to(dev)
        mm = torch.from_numpy((mask_metal > 0.5).astype(np.float32)[None, None]).to(dev)
        with torch.no_grad():
            x = m.reconstruct(b, l, mm)
        return x[0, 0].cpu().numpy()


def main():
    dev = "cuda"
    metas = [json.load(open(f)) for f in glob.glob(os.path.join(META, "*.json"))]
    ho = [m for m in metas if m["patient_id"] == "patients_head2"]
    print(f"[eval] held-out patients_head2 = {len(ho)} 样本", flush=True)

    # 预加载所有存在的模型
    models = {}
    for name, (path, mode) in W.items():
        if not os.path.exists(path):
            print(f"[skip] {name}: 权重不存在 {path}", flush=True)
            continue
        m, load, fwd_mode = load_model(mode, dev)
        sd = torch.load(path, map_location=dev, weights_only=False)
        if isinstance(sd, dict) and "find_net" in sd:
            sd = sd["find_net"]
        load(sd)
        models[name] = (m, fwd_mode)
        print(f"[load] {name}: {path}", flush=True)

    # 聚合：name -> category -> list[(mae,ssim,psnr)]
    agg = {name: defaultdict(list) for name in models}
    for i, meta in enumerate(ho):
        sid = meta["sample_id"]
        cat = meta["category"]
        base = rd(sid, "Baseline")
        tgt = rd(sid, "Target")
        li = rd(sid, "LI")
        msk = rd(sid, "Mask")
        mask_nm = (msk < 0.5).astype(np.float32)
        for name, (m, fwd_mode) in models.items():
            pred = run(m, fwd_mode, base, li, msk, dev)
            agg[name][cat].append(calc(tgt, pred, mask_nm))
        if (i + 1) % 120 == 0:
            print(f"  ...{i + 1}/{len(ho)}", flush=True)

    # 输出 2×2 表（按 category 分层 + 平均）
    order = ["giant", "large", "medium", "small"]
    print("\n=== MAE↓/SSIM↑/PSNR↑（masked, data_range=3000, held-out patients_head2）===", flush=True)
    for name in models:
        bycat = agg[name]
        allv = []
        print(f"\n[{name}]", flush=True)
        for cat in order:
            if cat not in bycat:
                continue
            v = bycat[cat]
            allv.extend(v)
            mm = tuple(np.mean(np.array(v), axis=0))
            print(f"  {cat:8s} n={len(v):3d} | MAE={mm[0]:6.2f} SSIM={mm[1]:.3f} PSNR={mm[2]:6.2f}", flush=True)
        if allv:
            mm = tuple(np.mean(np.array(allv), axis=0))
            print(f"  {'平均':8s} n={len(allv):3d} | MAE={mm[0]:6.2f} SSIM={mm[1]:.3f} PSNR={mm[2]:6.2f}", flush=True)


if __name__ == "__main__":
    main()
