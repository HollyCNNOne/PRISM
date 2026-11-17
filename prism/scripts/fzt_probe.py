#!/usr/bin/env python
"""fzt_probe.py —— 验证 FZT 核心假设：伪影高频 z 突变 vs 组织高频 z 连续。

对一条含金属序列，逐片算高频含量 H_z（梯度幅值），再算 z 轴连续性残差
R_z = |H_z − H_{z-1}|，对比「充分区（无伪影）」与「完全阴影区（有伪影）」的 R 均值：
  若 Baseline 里 R(阴影) >> R(充分)，且 Target 里两者接近 → FZT 假设成立。

纯 numpy，不训练，几分钟出结论。
"""
import glob
import json
import os

import numpy as np

D = "/root/autodl-tmp/PRISM/training_data_astra/train"
META = "/root/autodl-tmp/PRISM/training_data_astra/meta"
D_REF = 4.10023e+06


def rd(sid, sub):
    return np.fromfile(os.path.join(D, sub, f"{sid}.raw"), np.float32).reshape(512, 512)


def sample_idx(sid):
    return int(sid.rsplit("_", 1)[1])


def highfreq(x):
    """高频含量 = 中心差分梯度幅值（边缘强度）。"""
    gx = np.gradient(x, axis=1)
    gy = np.gradient(x, axis=0)
    return np.sqrt(gx ** 2 + gy ** 2)


def main():
    # 1. 按序列分组，找几条金属面积大的序列
    by_seq = {}
    for f in glob.glob(os.path.join(META, "*.json")):
        m = json.load(open(f))
        if m.get("split") == "train":
            by_seq.setdefault(m["sequence_id"], []).append(m["sample_id"])

    seqs = []
    for seq_id, sids in by_seq.items():
        if len(sids) < 6:            # 只要够长的序列（Z 足够，频域分辨够）
            continue
        sids = sorted(sids, key=sample_idx)
        metal_area = sum((rd(s, "Mask") > 0.5).sum() for s in sids)
        if metal_area > 500:         # 有足够金属
            seqs.append((metal_area, sids))
    seqs.sort(reverse=True)

    print(f"找到 {len(seqs)} 条含金属序列，取前 3 条", flush=True)
    for rank, (metal_area, sids) in enumerate(seqs[:3]):
        z = len(sids)
        print(f"\n===== 序列 #{rank} (Z={z}, 金属像素={metal_area}) =====", flush=True)
        # 2. 读整条序列
        base = np.stack([rd(s, "Baseline") for s in sids])          # [Z,H,W]
        tgt = np.stack([rd(s, "Target") for s in sids])
        msk = np.stack([rd(s, "Mask") for s in sids])
        dj = np.stack([np.load(os.path.join(D, "Phys", f"{s}_d_j.npy")).reshape(512, 512)
                       for s in sids])
        r = dj / D_REF

        # 3. 逐片高频含量 H_z
        Hb = np.stack([highfreq(base[i]) for i in range(z)])        # Baseline 高频
        Ht = np.stack([highfreq(tgt[i]) for i in range(z)])         # Target 高频

        # 4. z 轴连续性残差 R_z = |H_z - H_{z-1}|
        Rb = np.abs(np.diff(Hb, axis=0))                            # [Z-1,H,W]
        Rt = np.abs(np.diff(Ht, axis=0))

        # 5. 区域（排除金属核）
        nonmetal = msk < 0.5
        partial = nonmetal & (r >= 1.0 / 9.0) & (r <= 9.0)   # 部分阴影：伪影较弱
        shadow = nonmetal & (r < 1.0 / 9.0)                  # 完全阴影：伪影最强
        # r 是 [Z,H,W]，与 Rb [Z-1,H,W] 差一维：取 r[1:] 对齐 Rb
        partial_a = partial[1:]
        shadow_a = shadow[1:]

        def region_mean(R, reg):
            return float(R[reg].mean()) if reg.sum() > 0 else float("nan")

        rb_partial = region_mean(Rb, partial_a)
        rb_shadow = region_mean(Rb, shadow_a)
        rt_partial = region_mean(Rt, partial_a)
        rt_shadow = region_mean(Rt, shadow_a)
        print(f"  Baseline: 部分阴影 z-残差={rb_partial:.2f}  完全阴影 z-残差={rb_shadow:.2f}  "
              f"比值(完全/部分)={rb_shadow / rb_partial:.2f}", flush=True)
        print(f"  Target:   部分阴影 z-残差={rt_partial:.2f}  完全阴影 z-残差={rt_shadow:.2f}  "
              f"比值(完全/部分)={rt_shadow / rt_partial:.2f}", flush=True)
        print(f"  >> 伪影增量的 z-残差比 = Baseline完全阴影/Target完全阴影 = "
              f"{rb_shadow / rt_shadow:.2f}x", flush=True)


if __name__ == "__main__":
    main()
