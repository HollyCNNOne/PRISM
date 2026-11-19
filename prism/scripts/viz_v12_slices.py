#!/usr/bin/env python
"""viz_v12_slices.py —— 00_baseline 视觉验尸：held-out patients_head2 挑 3 张代表性切片。

抽样：
  1. 重度伪影过渡区：complete/large 线圈、金属面积大 + 星芒强（高残差方差）+ 金属近画面中心。
  2. 复杂解剖区：非金属区骨密度占比最高（颅底/牙齿，HU>300）。
  3. 纯组织区：金属面积≈0、脑组织（HU∈[0,150]）占比最高（验证充分区 RMSE≈33）。

每片：跑其所在序列的完整双向递归（SSM 需邻居上下文），取该片 x̂_final。
出 4 列图：Target | Baseline | Prediction | Error(hot+标注 max)；前三列强制 vmin=-150/vmax=350。
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/autodl-tmp/prism_framework")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prism.data.dataset import make_theta_sequence_loader  # noqa: E402
from prism.models.m_network import MNetwork  # noqa: E402
from prism.models.find_net import FINDNet, FINDNetArgs  # noqa: E402
from prism.recursion.algorithm1 import forward_backward  # noqa: E402
from prism.training.trainer import SSMTrainer  # noqa: E402
from prism.utils.config import SSMConfig  # noqa: E402


def _sample_idx(sid):
    return int(sid.rsplit("_", 1)[1])


def rd(path, h=512, w=512):
    return np.fromfile(path, dtype=np.float32).reshape(h, w)


def highpass_std(x):
    """x:[H,W] numpy → 高频残差 std（星芒强度代理）。"""
    t = torch.from_numpy(x.astype(np.float32))[None, None]
    smooth = F.avg_pool2d(t, 5, stride=1, padding=2)
    return (t - smooth).std().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="/root/autodl-tmp/PRISM/training_data_astra/ablations_v12/00_baseline.pt")
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data_astra")
    ap.add_argument("--patient", default="patients_head2")
    ap.add_argument("--out-dir", default="/root/autodl-tmp/PRISM/training_data_astra/viz_v12")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    meta_dir = os.path.join(args.data_dir, "meta")
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 1. 扫描 patients_head2，选 3 片 ----
    scored = []
    for f in glob.glob(os.path.join(meta_dir, "*.json")):
        m = json.load(open(f))
        if m.get("patient_id") != args.patient:
            continue
        sid = m["sample_id"]
        base = rd(os.path.join(train_dir, "Baseline", f"{sid}.raw"))
        mask = rd(os.path.join(train_dir, "Mask", f"{sid}.raw"))
        tgt = rd(os.path.join(train_dir, "Target", f"{sid}.raw"))
        binm = mask > 0.5
        metal_area = float(binm.sum())
        nonm = ~binm
        n_nonm = max(int(nonm.sum()), 1)
        center_dist = 1e9
        if metal_area > 0:
            ys, xs = np.nonzero(binm)
            center_dist = float(np.hypot(ys.mean() - 256, xs.mean() - 256))
        tissue = (tgt >= 0) & (tgt <= 150) & nonm
        bone = (tgt > 300) & nonm
        tissue_frac = float(tissue.sum()) / n_nonm
        bone_frac = float(bone.sum()) / n_nonm
        streak = highpass_std(base * nonm.astype(np.float32))
        scored.append(dict(sid=sid, cat=m.get("case_category", ""), seq=m.get("sequence_id"),
                           metal_area=metal_area, center_dist=center_dist,
                           tissue_frac=tissue_frac, bone_frac=bone_frac, streak=streak,
                           base=base, mask=mask, tgt=tgt))

    big = [s for s in scored if s["metal_area"] > 50]
    heavy = sorted(big, key=lambda s: (-s["metal_area"], -s["streak"], s["center_dist"]))[0]
    complex_ = sorted(scored, key=lambda s: -s["bone_frac"])[0]
    pure = sorted([s for s in scored if s["metal_area"] < 5],
                  key=lambda s: (-s["tissue_frac"], s["metal_area"]))[0]

    picks = [("heavy", heavy, "Heavy-artifact transition zone"),
             ("complex", complex_, "Complex anatomy (skull base)"),
             ("pure", pure, "Pure tissue (far from metal)")]

    for name, s, label in picks:
        print(f"[pick {name}] {s['sid']} cat={s['cat']} metal_area={s['metal_area']:.0f} "
              f"center_dist={s['center_dist']:.0f} bone_frac={s['bone_frac']:.3f} "
              f"tissue_frac={s['tissue_frac']:.3f} streak={s['streak']:.1f}", flush=True)

    # ---- 2. 载入模型 ----
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = ckpt["config"]
    d_ref = cfg["d_ref"]
    nb = cfg.get("nb", 1)
    channels = cfg.get("channels", 32)
    use_velocity = cfg.get("use_velocity", True)
    recursion = cfg.get("recursion", "bidirectional")
    gain_mode = cfg.get("gain_mode", "kalman")
    findnet_lict_source = cfg.get("findnet_lict_source", "mu")
    binarize_mask = cfg.get("binarize_mask", False)
    mask_bin_thr = cfg.get("mask_bin_thr", 0.5)

    m_net = MNetwork(nb=nb, channels=channels, use_velocity=use_velocity).to(args.device)
    find_net = FINDNet(FINDNetArgs()).to(args.device)
    # FIND-Net 的 etaM_S/etaX_S/K_q/tau 是 expand 视图（stride 0），CPU 上 copy_ 会报
    # "more than one element refers to a single memory location"，用 assign=True 直接替换 data。
    m_net.load_state_dict(ckpt["model"], assign=True)
    find_net.load_state_dict(ckpt["find_net"], assign=True)
    m_net.eval()
    find_net.eval()
    config = SSMConfig(nb=nb, channels=channels, recursion=recursion, d_ref=d_ref,
                       gain_mode=gain_mode, findnet_lict_source=findnet_lict_source,
                       binarize_mask=binarize_mask, mask_bin_thr=mask_bin_thr)
    trainer = SSMTrainer(m_net, config, device=args.device, find_net=find_net)
    loader = make_theta_sequence_loader(
        os.path.join(train_dir, "Target"), os.path.join(train_dir, "Baseline"),
        os.path.join(train_dir, "LI"), os.path.join(train_dir, "Mask"),
        os.path.join(train_dir, "Phys"))

    # ---- 3. 按序列去重跑推理（缓存 x_hat_final）----
    by_seq = {}
    for s in scored:
        by_seq.setdefault(s["seq"], []).append(s["sid"])

    def run_seq(seq):
        sample = loader(seq)
        thetas = [{k: v.unsqueeze(0).to(args.device) for k, v in t.items()} for t in sample["thetas"]]
        x_hat = sample["x_hat"].unsqueeze(0).to(args.device)
        X_clean = sample["X_clean"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(args.device)
        metal_mask = sample["metal_mask"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(args.device)
        li = sample["li"].unsqueeze(0).to(args.device)
        Z = len(thetas)
        dirty_vol = x_hat.permute(0, 2, 1, 3, 4)
        dirty = [dirty_vol[:, :, z] for z in range(Z)]
        li_vol = li.permute(0, 2, 1, 3, 4)
        li_list = [li_vol[:, :, z] for z in range(Z)]
        mask_input = (metal_mask if not config.binarize_mask
                      else (metal_mask > config.mask_bin_thr).float())
        masks = [mask_input[:, :, z] for z in range(Z)]
        observer = trainer._observer(dirty, masks, li_list)
        with torch.no_grad():
            res = forward_backward(m_net, thetas, d_ref, recursion=recursion,
                                   observer=observer, gain_mode=config.gain_mode)
            x_hat_final = trainer._final_output(res, dirty, masks, li_list)  # [1,1,Z,H,W]
        return seq, x_hat_final

    cache = {}
    for name, s, label in picks:
        seq = sorted(by_seq[s["seq"]], key=_sample_idx)
        if s["seq"] not in cache:
            cache[s["seq"]] = run_seq(seq)
        _, x_hat_final = cache[s["seq"]]
        zi = seq.index(s["sid"])
        pred = x_hat_final[0, 0, zi].cpu().numpy()
        tgt = s["tgt"]
        base = s["base"]
        err = np.abs(pred - tgt)
        nonm = ~(s["mask"] > 0.5)
        rmse = float(np.sqrt(np.mean((pred[nonm] - tgt[nonm]) ** 2)))

        fig, axes = plt.subplots(1, 4, figsize=(24, 6.3))
        for ax, img, t in [(axes[0], tgt, "Target (GT)"),
                           (axes[1], base, "Baseline (artifact)"),
                           (axes[2], pred, "Prediction (PRISM)")]:
            ax.imshow(img, cmap="gray", vmin=-150, vmax=350, interpolation="nearest")
            ax.set_title(t, fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
        im3 = axes[3].imshow(err, cmap="hot", interpolation="nearest")
        axes[3].set_title(f"Error |pred-tgt|  max={err.max():.0f} HU", fontsize=14)
        axes[3].set_xticks([])
        axes[3].set_yticks([])
        fig.colorbar(im3, ax=axes[3], fraction=0.046)
        fig.suptitle(
            f"v1.2-00_baseline | {label} | {s['sid']} (cat={s['cat']})\n"
            f"non-metal RMSE={rmse:.1f} HU | pred min/max/mean={pred.min():.0f}/{pred.max():.0f}/{pred.mean():.0f} "
            f"| tgt mean={tgt.mean():.0f}", fontsize=12)
        out = os.path.join(args.out_dir, f"v12_00baseline_{name}_{s['sid']}.png")
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"[save] {out}  (non-metal RMSE={rmse:.1f} HU)", flush=True)


if __name__ == "__main__":
    main()
