#!/usr/bin/env python
"""z_fsp_fig.py —— Z-FSP 论文图（分开展示，全局整片不 zoom）。

图 1（montage）：全局整片沿 z 的蒙太奇——Target 一行、Baseline 一行，每行把全部
  Z 片横向拼接，框出 3 个特征感受野（金属核/条纹/组织）。一眼看「干净组织沿 z 连续
  漂移」vs「金属/条纹沿 z 突然出现-消失」。
图 2（signal）：每个感受野的 1D 信号曲线 + 1D FFT 幅频。
"""
import glob
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/root/autodl-tmp/PRISM/training_data_astra/train"
META = "/root/autodl-tmp/PRISM/training_data_astra/meta"
D_REF = 4.10023e+06
DS = 2            # 下采样因子（512→256）
P = 18            # patch 半宽（原分辨率）


def rd(sid, sub):
    return np.fromfile(os.path.join(D, sub, f"{sid}.raw"), np.float32).reshape(512, 512)


def sample_idx(sid):
    return int(sid.rsplit("_", 1)[1])


def main():
    by_seq = {}
    for f in glob.glob(os.path.join(META, "*.json")):
        m = json.load(open(f))
        if m.get("split") == "train":
            by_seq.setdefault(m["sequence_id"], []).append(m["sample_id"])

    best = None
    for seq_id, sids in by_seq.items():
        if len(sids) < 8:
            continue
        sids = sorted(sids, key=sample_idx)
        area = sum((rd(s, "Mask") > 0.5).sum() for s in sids)
        if best is None or area > best[0]:
            best = (area, sids)
    _, sids = best
    Z = len(sids)
    print(f"序列 Z={Z}，金属像素={best[0]}", flush=True)

    tgt = np.stack([rd(s, "Target") for s in sids])       # [Z,H,W]
    base = np.stack([rd(s, "Baseline") for s in sids])
    msk = np.stack([rd(s, "Mask") for s in sids])
    dj = np.stack([np.load(os.path.join(D, "Phys", f"{s}_d_j.npy")).reshape(512, 512)
                   for s in sids])

    zi = Z // 2
    mid_mask = msk[zi] > 0.5
    ys, xs = np.nonzero(mid_mask)
    cy, cx = int(ys.mean()), int(xs.mean())

    r = dj[zi] / D_REF
    yy, xx = np.mgrid[0:512, 0:512]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    # 组织区：固定取金属左上偏移（可调），远离条纹（右侧）与颅骨
    ty, tx = int(min(max(cy - 60, P), 511 - P)), int(min(max(cx - 40, P), 511 - P))
    print(f"  Tissue 框位置=({ty},{tx})，该点 HU={tgt[zi, ty, tx]:.0f}，d_j/d_ref={r[ty, tx]:.2f}", flush=True)

    def _c(v):
        return int(min(max(v, P), 511 - P))

    patches = [
        ("Metal core", _c(cy), _c(cx), "lime"),
        ("Streak", _c(cy), _c(cx + 35), "red"),
        ("Tissue", _c(ty), _c(tx), "cyan"),
    ]

    # ---- 图 1：全局整片沿 z 的蒙太奇 ----
    def montage(vol):
        return np.concatenate([vol[z, ::DS, ::DS] for z in range(Z)], axis=1)  # [256, Z*256]

    mt = montage(tgt)
    mb = montage(base)
    Hd = 512 // DS  # 256

    fig, (ax_t, ax_b) = plt.subplots(2, 1, figsize=(Z * 3.2, 6.4))
    for ax, img, title in [(ax_t, mt, "Clean (Target) — full slices along z"),
                           (ax_b, mb, "Artifact (Baseline) — full slices along z")]:
        ax.imshow(img, cmap="gray", vmin=-150, vmax=350)
        for z in range(1, Z):
            ax.axvline(z * Hd - 0.5, color="white", linewidth=0.6, alpha=0.5)
        for (name, py, px, c) in patches:
            # 框出感受野（每个 z 片都画一个小框，缩放到下采样尺寸）
            bsz = P * 2 // DS
            for z in range(Z):
                ax.add_patch(plt.Rectangle((z * Hd + px // DS - bsz // 2,
                                            py // DS - bsz // 2), bsz, bsz,
                                           fill=False, edgecolor=c, linewidth=1.3))
        ax.set_xticks([(z + 0.5) * Hd for z in range(Z)])
        ax.set_xticklabels([f"{z}" for z in range(Z)], fontsize=8)
        ax.set_xlabel("z (slice)")
        ax.set_ylabel("y")
        ax.set_title(title, fontsize=12)

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor="none", edgecolor=c, label=n) for (n, _, _, c) in patches],
               loc="upper center", ncol=3, fontsize=9)
    fig.suptitle("Z-FSP: global slices evolve along z — clean drifts smoothly, metal appears abruptly",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out1 = "/root/autodl-tmp/PRISM/training_data_astra/viz_v12/z_fsp_montage.png"
    os.makedirs(os.path.dirname(out1), exist_ok=True)
    fig.savefig(out1, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out1}", flush=True)

    # ---- 图 2：1D 信号（Target/Baseline 分行，各自 y 轴）+ 频谱（log 幅值）----
    def zsig(vol, py, px):
        s = vol[:, py - P:py + P + 1, px - P:px + P + 1]
        return s.reshape(Z, -1).mean(axis=1)

    def _yrange(s):
        lo, hi = float(s.min()), float(s.max())
        if hi - lo < 1e-6:
            pad = max(abs(hi) * 0.1, 1.0)
        else:
            pad = (hi - lo) * 0.15
        return lo - pad, hi + pad

    freqs = np.fft.rfftfreq(Z)
    fig2, axes = plt.subplots(3, 3, figsize=(15, 13))
    for j, (name, py, px, c) in enumerate(patches):
        st = zsig(tgt, py, px)
        sb = zsig(base, py, px)

        # Row 0: Target 单独（自有 y 轴，不被金属阶跃压扁）
        ax = axes[0, j]
        ax.plot(range(Z), st, "o-", color="tab:blue", markersize=4)
        ax.set_title(f"{name} — Target z signal", fontsize=11)
        ax.set_ylabel("HU")
        ax.set_ylim(_yrange(st))

        # Row 1: Baseline 单独（自有 y 轴，金属阶跃完整可见）
        ax = axes[1, j]
        ax.plot(range(Z), sb, "s-", color="tab:red", markersize=4)
        ax.set_title(f"{name} — Baseline z signal", fontsize=11)
        ax.set_ylabel("HU")
        ax.set_ylim(_yrange(sb))

        # Row 2: 频谱 log 幅值（同轴，形状=宽带 vs 集中）
        ax = axes[2, j]
        Ft = np.abs(np.fft.rfft(st - st.mean()))
        Fb = np.abs(np.fft.rfft(sb - sb.mean()))
        ax.plot(freqs, np.log10(1.0 + Ft), "o-", color="tab:blue", markersize=4)
        ax.plot(freqs, np.log10(1.0 + Fb), "s-", color="tab:red", markersize=4)
        ax.set_title(f"{name} — |FFT| (log10)", fontsize=11)
        ax.set_xlabel("freq (cycles/slice)")
        ax.set_ylabel("log10(1+|FFT|)")

    from matplotlib.lines import Line2D
    fig2.legend(handles=[
        Line2D([0], [0], color="tab:blue", marker="o", linestyle="-", label="Target"),
        Line2D([0], [0], color="tab:red", marker="s", linestyle="-", label="Baseline"),
    ], loc="upper center", ncol=2, fontsize=9)
    fig2.suptitle("Z-FSP: z-axis 1D signal (split y-axis) & log spectrum", fontsize=13)
    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    out2 = "/root/autodl-tmp/PRISM/training_data_astra/viz_v12/z_fsp_signal.png"
    fig2.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"[save] {out2}", flush=True)


if __name__ == "__main__":
    main()
