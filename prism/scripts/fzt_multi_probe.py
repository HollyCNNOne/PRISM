#!/usr/bin/env python
"""fzt_multi_probe.py —— 跨序列 FZT 验证：伪影 z 突变是否普遍成立。

对多条含金属序列，算「部分阴影区（条纹带，r∈[1/9,9]）」的高频 z 轴不连续度：
  R = mean_z |H_z − H_{z-1}|（高频=梯度幅值）
  伪影信号比 = Baseline_R / Target_R（>1 表示伪影确实在 z 轴更突变）
画柱状图：每条序列一个比值，证明 FZT 假设跨序列普遍成立。
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


def rd(sid, sub):
    return np.fromfile(os.path.join(D, sub, f"{sid}.raw"), np.float32).reshape(512, 512)


def sample_idx(sid):
    return int(sid.rsplit("_", 1)[1])


def highfreq(x):
    gx = np.gradient(x, axis=1)
    gy = np.gradient(x, axis=0)
    return np.sqrt(gx ** 2 + gy ** 2)


def zdiscont(vol, mask):
    """vol [Z,H,W] 在 mask 内的 z 轴不连续度 = mean |H_z − H_{z-1}|。"""
    H = np.stack([highfreq(vol[i]) for i in range(vol.shape[0])])
    R = np.abs(np.diff(H, axis=0))
    m = mask[1:]  # 对齐 R [Z-1]
    return float(R[m].mean()) if m.sum() > 0 else float("nan")


def main():
    by_seq = {}
    for f in glob.glob(os.path.join(META, "*.json")):
        m = json.load(open(f))
        if m.get("split") == "train":
            by_seq.setdefault(m["sequence_id"], []).append(m["sample_id"])

    rows = []
    for seq_id, sids in by_seq.items():
        if len(sids) < 6:
            continue
        sids = sorted(sids, key=sample_idx)
        msk = np.stack([rd(s, "Mask") for s in sids])
        if (msk > 0.5).sum() < 300:      # 要有足够金属
            continue
        base = np.stack([rd(s, "Baseline") for s in sids])
        tgt = np.stack([rd(s, "Target") for s in sids])
        dj = np.stack([np.load(os.path.join(D, "Phys", f"{s}_d_j.npy")).reshape(512, 512)
                       for s in sids])
        r = dj / D_REF
        nonmetal = msk < 0.5
        partial = nonmetal & (r >= 1.0 / 9.0) & (r <= 9.0)   # 条纹带
        if partial.sum() < 1000:
            continue
        rb = zdiscont(base, partial)
        rt = zdiscont(tgt, partial)
        if rb == rb and rt == rt and rt > 1e-3:
            rows.append((len(sids), float((msk > 0.5).sum()), rb, rt, rb / rt))
        if len(rows) >= 8:
            break

    print(f"跨序列 FZT 验证：{len(rows)} 条序列", flush=True)
    for z, ma, rb, rt, ratio in rows:
        print(f"  Z={z:2d} 金属={ma:5.0f}  Baseline_R={rb:6.2f}  Target_R={rt:6.2f}  比值={ratio:.2f}x",
              flush=True)

    ratios = [x[4] for x in rows]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = np.arange(len(ratios))
    bars = ax.bar(xs, ratios, color="tab:red", alpha=0.85)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="ratio=1 (no artifact)")
    for x, ratio in zip(xs, ratios):
        ax.text(x, ratio + 0.03, f"{ratio:.2f}", ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"seq{i + 1}" for i in range(len(ratios))])
    ax.set_xlabel("sequence")
    ax.set_ylabel("z-discontinuity ratio (Baseline / Target)")
    ax.set_title("FZT: artifact adds z-axis discontinuity (partial-shadow region) — "
                 f"mean={np.mean(ratios):.2f}x across {len(ratios)} sequences", fontsize=12)
    ax.legend()
    ax.set_ylim(0, max(ratios) * 1.2)
    fig.tight_layout()
    out = "/root/autodl-tmp/PRISM/training_data_astra/viz_v12/fzt_multi_probe.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[save] {out}", flush=True)


if __name__ == "__main__":
    main()
