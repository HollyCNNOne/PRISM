#!/usr/bin/env python
"""check_streak.py —— Stage 1 条纹压制定量检验（无视觉依赖，数字说话）。

对 4 种线圈尺寸各挑验证集「条纹最重」切片（判据同 viz_stage1：非金属区 (Baseline−Target)²
最大），然后量两类区域的 RMSE（HU）：
    - 非金属全区（1 − metal_core）
    - 过渡区（金属核外扩 10px 环带，条纹最凶的地带）
每个区域各报 Baseline / Prediction 两条 RMSE + 压制比（baseline/pred，越大=压得越干净）。
Stage 1 是纯 2D FIND-Net，只验收「放射状条纹有没有被擦掉」，不评牙齿/骨骼/整体 HU。
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

from prism.models.find_net import FINDNet, FINDNetArgs  # noqa: E402

SIZES = ["small", "medium", "large", "giant"]


def _read_np(path, h=512, w=512):
    return np.fromfile(path, dtype=np.float32).reshape(h, w)


def ring_mask(msk_np, dilate_px=10, thr=0.01):
    """金属核外扩环带（过渡区），返回 [H,W] 0/1。"""
    binm = torch.from_numpy((msk_np > thr).astype(np.float32))[None, None]  # [1,1,H,W]
    k = 2 * dilate_px + 1
    dilated = F.max_pool2d(binm, kernel_size=k, stride=1, padding=dilate_px)
    ring = (dilated - binm).clamp(min=0.0)
    return ring.squeeze().numpy()


def rmse(arr, region):
    return float(np.sqrt(((arr) ** 2 * region).sum() / (region.sum() + 1e-9)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data_astra")
    ap.add_argument("--val-patient", default="patients_head2")
    ap.add_argument("--dilate-px", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    find_net = FINDNet(FINDNetArgs()).to(args.device)
    find_net.load_state_dict(ck["find_net"])
    find_net.eval()
    ep = ck.get("config", {}).get("epoch", "?")
    print(f"[load] {args.checkpoint} (epoch {ep})\n", flush=True)

    train_dir = os.path.join(args.data_dir, "train")
    metas = {}
    for f in glob.glob(os.path.join(args.data_dir, "meta", "*.json")):
        m = json.load(open(f))
        if m.get("patient_id") == args.val_patient:
            metas[m["sample_id"]] = m

    hdr = f"{'size':<7}{'sample':<15}{'Base_NM':>10}{'Pred_NM':>10}{'sup_NM':>8}" \
          f"{'Base_TZ':>10}{'Pred_TZ':>10}{'sup_TZ':>8}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for size in SIZES:
        sids = [sid for sid, m in metas.items() if m.get("category") == size]
        best_sid, best_e = None, -1.0
        for sid in sids:
            base = _read_np(os.path.join(train_dir, "Baseline", f"{sid}.raw"))
            tgt = _read_np(os.path.join(train_dir, "Target", f"{sid}.raw"))
            msk = _read_np(os.path.join(train_dir, "Mask", f"{sid}.raw"))
            w = 1.0 - (msk > 0.01)
            e = float(((base - tgt) ** 2 * w).sum() / (w.sum() + 1e-9))
            if e > best_e:
                best_e, best_sid = e, sid

        base = _read_np(os.path.join(train_dir, "Baseline", f"{best_sid}.raw"))
        lict = _read_np(os.path.join(train_dir, "LI", f"{best_sid}.raw"))
        msk = _read_np(os.path.join(train_dir, "Mask", f"{best_sid}.raw"))
        tgt = _read_np(os.path.join(train_dir, "Target", f"{best_sid}.raw"))
        with torch.no_grad():
            pred = find_net.reconstruct(
                torch.from_numpy(base).float()[None, None].to(args.device),
                torch.from_numpy(lict).float()[None, None].to(args.device),
                torch.from_numpy(msk).float()[None, None].to(args.device),
            ).squeeze().cpu().numpy()

        nm = 1.0 - (msk > 0.01)                      # 非金属全区
        tz = ring_mask(msk, args.dilate_px)          # 过渡区环带
        b_nm = rmse(base - tgt, nm)
        p_nm = rmse(pred - tgt, nm)
        b_tz = rmse(base - tgt, tz)
        p_tz = rmse(pred - tgt, tz)
        sup_nm = b_nm / max(p_nm, 1e-6)
        sup_tz = b_tz / max(p_tz, 1e-6)
        print(f"{size:<7}{best_sid:<15}{b_nm:>10.1f}{p_nm:>10.1f}{sup_nm:>8.1f}x"
              f"{b_tz:>10.1f}{p_tz:>10.1f}{sup_tz:>8.1f}x", flush=True)


if __name__ == "__main__":
    main()
