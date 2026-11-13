#!/usr/bin/env python
"""make_b_ablation_fig.py —— 生成 B 递归消融对比图（同一 held-out 切片，多臂并排）。

找 patients_head2 里「完全阴影（r=d_j/d_ref<1/9，非金属）体素最多」的那张切片，把该切片的
输入 / 真值 / 完全阴影高亮 / per_slice(无递归) / 双向+速度 / 双向去速度(定稿) 并排成一张图。
各臂完全阴影 fin_RMSE 已由 eval_mar.py 定案（1678 / 616 / 306 HU），本脚本只出图不做数值评测。
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_FRAMEWORK_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _FRAMEWORK_ROOT)

from prism.data.dataset import make_theta_sequence_loader  # noqa: E402
from prism.models.m_network import MNetwork  # noqa: E402
from prism.models.find_net import FINDNet, FINDNetArgs  # noqa: E402
from prism.recursion.algorithm1 import forward_backward  # noqa: E402
from prism.training.trainer import SSMTrainer  # noqa: E402
from prism.utils.config import SSMConfig  # noqa: E402


def _sample_idx(sid: str) -> int:
    return int(sid.rsplit("_", 1)[1])


def build_patient_specs(meta_dir: str, patient_id: str, min_z: int = 3, max_z: int = 0):
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
    return specs


def load_arm(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    d_ref = cfg["d_ref"]
    nb = cfg.get("nb", 1)
    channels = cfg.get("channels", 32)
    m_net = MNetwork(nb=nb, channels=channels,
                     use_velocity=cfg.get("use_velocity", True),
                     gate_reads_dj=cfg.get("gate_reads_dj", False),
                     ablate_xm0=cfg.get("ablate_xm0", False),
                     ablate_dj=cfg.get("ablate_dj", False),
                     ablate_g=cfg.get("ablate_g", False)).to(device)
    find_net = FINDNet(FINDNetArgs()).to(device)
    m_net.load_state_dict(ckpt["model"])
    find_net.load_state_dict(ckpt["find_net"])
    m_net.eval()
    find_net.eval()
    config = SSMConfig(nb=nb, channels=channels, trust_mode="kalman", d_ref=d_ref)
    trainer = SSMTrainer(m_net, config, device=device, find_net=find_net)
    return m_net, find_net, trainer, d_ref


def run_arm(trainer, m_net, find_net, d_ref, recursion, thetas, x_hat, metal_mask):
    Z = len(thetas)
    dirty_vol = x_hat.permute(0, 2, 1, 3, 4)                    # [1,1,Z,H,W]
    dirty = [dirty_vol[:, :, z] for z in range(Z)]              # [Z] 个 [1,1,H,W]
    masks = [metal_mask[:, :, z] for z in range(Z)]             # [Z] 个 [1,1,H,W]
    observer = trainer._observer(dirty, masks)
    res = forward_backward(m_net, thetas, d_ref, recursion=recursion, observer=observer)
    x_hat_final = trainer._final_output(res, dirty, masks)      # [1,1,Z,H,W]
    return x_hat_final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data")
    ap.add_argument("--patient", default="patients_head2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="/root/autodl-tmp/prism_framework/b_ablation_fig")
    ap.add_argument("--vmin", type=float, default=-200.0)
    ap.add_argument("--vmax", type=float, default=400.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_dir = os.path.join(args.data_dir, "train")
    meta_dir = os.path.join(args.data_dir, "meta")
    loader = make_theta_sequence_loader(
        os.path.join(train_dir, "Target"), os.path.join(train_dir, "Baseline"),
        os.path.join(train_dir, "LI"), os.path.join(train_dir, "Mask"),
        os.path.join(train_dir, "Phys"))

    # d_ref 所有臂一致（held-out 冻结 = 1.23673e9），拿一个 checkpoint 读即可（CPU，不占 GPU）
    first_ckpt = os.path.join(args.data_dir, "ssm_nb1_kalman_ho_patients_head2_b1_novel_gate.pt")
    _, _, _, d_ref = load_arm(first_ckpt, "cpu")

    specs = build_patient_specs(meta_dir, args.patient)

    # 找完全阴影（r<1/9 且非金属）体素最多的 (spec, slice)，保留该样本在内存
    best = None  # (n_shadow, spec, z, sample)
    for spec in specs:
        sample = loader(spec)
        for z in range(len(sample["thetas"])):
            dj = sample["thetas"][z]["d_j"]          # [1,H,W] torch (CPU)
            metal = sample["metal_mask"][z]          # [1,H,W]
            r = dj / d_ref
            n_shadow = int(((r < 1.0 / 9.0) & (metal < 0.5)).sum().item())
            if best is None or n_shadow > best[0]:
                best = (n_shadow, spec, z, sample)
    n_shadow, spec, z_sel, sample = best
    print(f"[pick] 完全阴影体素 {n_shadow} 的切片: spec[0]={spec[0]} 序列 {len(spec)} 片, "
          f"slice={z_sel}", flush=True)

    # 移到 device
    device = args.device
    thetas = [{k: v.unsqueeze(0).to(device) for k, v in t.items()} for t in sample["thetas"]]
    x_hat = sample["x_hat"].unsqueeze(0).to(device)                    # [1,Z,1,H,W]
    X_clean = sample["X_clean"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)  # [1,1,Z,H,W]
    metal_mask = sample["metal_mask"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)

    ARMS = [
        ("per_slice (no recursion)\n1678 HU", "ssm_nb1_kalman_ho_patients_head2_b8_norec.pt", "per_slice"),
        ("bidir + velocity\n616 HU", "ssm_nb1_kalman_ho_patients_head2_b8_shadow.pt", "bidirectional"),
        ("bidir no-velocity (final)\n306 HU", "ssm_nb1_kalman_ho_patients_head2_b1_novel_gate.pt", "bidirectional"),
    ]

    outs = {}
    for label, ckpt_name, recursion in ARMS:
        ckpt_path = os.path.join(args.data_dir, ckpt_name)
        m_net, find_net, trainer, _d = load_arm(ckpt_path, device)
        assert abs(_d - d_ref) < 1e-3, f"d_ref mismatch: {_d} vs {d_ref}"
        with torch.no_grad():
            x_hat_final = run_arm(trainer, m_net, find_net, d_ref, recursion,
                                  thetas, x_hat, metal_mask)
        outs[label] = x_hat_final[0, 0, z_sel].detach().cpu().numpy()   # [H,W]
        print(f"[arm] {ckpt_name} recursion={recursion} done", flush=True)

    # 参考图
    x_in = x_hat[0, z_sel, 0].detach().cpu().numpy()        # [H,W] 输入
    x_gt = X_clean[0, 0, z_sel].detach().cpu().numpy()      # [H,W] 真值
    d_j = thetas[z_sel]["d_j"]                              # [1,1,H,W]
    r_map = (d_j / d_ref)[0, 0].detach().cpu().numpy()      # [H,W]
    metal = metal_mask[0, 0, z_sel].detach().cpu().numpy()  # [H,W]
    shadow = (r_map < 1.0 / 9.0) & (metal < 0.5)           # 完全阴影（非金属）

    vmin, vmax = args.vmin, args.vmax
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    def im(ax, arr, title, cmap="gray"):
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    im(axes[0, 0], x_in, f"Input x-hat (artifact)\nsample {spec[0]} slice {z_sel}")
    im(axes[0, 1], x_gt, "Ground truth")
    im(axes[0, 2], x_gt, "Shadow zone (r<1/9)")
    axes[0, 2].imshow(shadow, cmap="Blues", alpha=0.55)

    labels = list(outs.keys())
    im(axes[1, 0], outs[labels[0]], labels[0])
    im(axes[1, 1], outs[labels[1]], labels[1])
    im(axes[1, 2], outs[labels[2]], labels[2])

    fig.suptitle(f"B ablation: recursion fills the complete-shadow zone  "
                 f"(window [{vmin},{vmax}] HU, held-out {args.patient})", fontsize=13)
    out_path = os.path.join(args.out_dir, f"b_ablation_{spec[0]}_z{z_sel}.png")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}", flush=True)


if __name__ == "__main__":
    main()
