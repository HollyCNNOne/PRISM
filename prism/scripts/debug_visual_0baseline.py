#!/usr/bin/env python
"""debug_visual_0baseline.py — 视觉+张量双重诊断 00_baseline「充分区 RMSE 995」。

复用 eval_mar.py 的加载/递归/分区逻辑（与真实评测逐字对齐），额外：
  1) 充分区（d_j/d_ref>9，非金属）打印 Prediction/Target 的 min/max/mean（反归一化诊断）；
  2) 挑 3 张典型切片（纯组织/过渡/重伪影），画 Target|Baseline|Prediction|Error 四列，
     固定 vmin=-150 vmax=350（Error 用 0..350 热力）。

用法：
  python scripts/debug_visual_0baseline.py --checkpoint <...>/00_baseline.pt \
    --data-dir /root/autodl-tmp/PRISM/training_data_astra --patient patients_head2 \
    --out-dir /root/autodl-tmp/PRISM/debug_0baseline
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

_FRAMEWORK_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _FRAMEWORK_ROOT)

from prism.data.dataset import make_theta_sequence_loader
from prism.models.m_network import MNetwork
from prism.models.find_net import FINDNet, FINDNetArgs
from prism.recursion.algorithm1 import forward_backward
from prism.training.trainer import SSMTrainer
from prism.utils.config import SSMConfig


def _sample_idx(sid: str) -> int:
    return int(sid.rsplit("_", 1)[1])


def build_patient_specs(meta_dir, patient_id, min_z=3, max_z=0):
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


def transition_zone_mask(metal_mask, r, dilate_px=10, r_low=1.0 / 9.0):
    binm = (metal_mask > 0.5).float()
    k = 2 * dilate_px + 1
    dilated = F.max_pool3d(binm, kernel_size=(1, k, k), stride=1,
                           padding=(0, dilate_px, dilate_px))
    ring = (dilated - binm).clamp(min=0.0)
    low = (r < r_low).float()
    return ring * low


def _new_st():
    return {"n": 0, "sum": 0.0, "ss": 0.0, "mn": float("inf"), "mx": float("-inf")}


def _acc(st, v):
    if v.numel() == 0:
        return
    st["n"] += int(v.numel())
    st["sum"] += float(v.sum())
    st["ss"] += float((v * v).sum())
    st["mn"] = min(st["mn"], float(v.min()))
    st["mx"] = max(st["mx"], float(v.max()))


def _better(pick, score, data):
    if pick is None or score > pick["score"]:
        return {"score": score, **data}
    return pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data_astra")
    ap.add_argument("--patient", required=True)
    ap.add_argument("--trust-mode", default="kalman")
    ap.add_argument("--recursion", default="bidirectional")
    ap.add_argument("--nb", type=int, default=1)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-z", type=int, default=3)
    ap.add_argument("--max-z", type=int, default=0)
    ap.add_argument("--max-implants", type=int, default=0)
    ap.add_argument("--gain-mode", default="kalman")
    ap.add_argument("--findnet-lict", default="mu")
    ap.add_argument("--out-dir", default="/root/autodl-tmp/PRISM/debug_0baseline")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

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
    findnet_lict = cfg.get("findnet_lict_source", args.findnet_lict)
    binarize = cfg.get("binarize_mask", False)
    bin_thr = cfg.get("mask_bin_thr", 0.5)
    recursion = cfg.get("recursion", args.recursion)

    m_net = MNetwork(nb=nb, channels=channels, use_velocity=use_velocity,
                     gate_reads_dj=gate_reads_dj, ablate_xm0=ablate_xm0,
                     ablate_dj=ablate_dj, ablate_g=ablate_g).to(args.device)
    find_net = FINDNet(FINDNetArgs()).to(args.device)
    m_net.load_state_dict(ckpt["model"])
    find_net.load_state_dict(ckpt["find_net"])
    m_net.eval()
    find_net.eval()

    config = SSMConfig(nb=nb, channels=channels, trust_mode=args.trust_mode,
                       recursion=recursion, d_ref=d_ref, gain_mode=gain_mode,
                       findnet_lict_source=findnet_lict, binarize_mask=binarize,
                       mask_bin_thr=bin_thr)
    trainer = SSMTrainer(m_net, config, device=args.device, find_net=find_net)

    print(f"[debug] ckpt={args.checkpoint}\n"
          f"  patient={args.patient} nb={nb} d_ref={d_ref:.6g} recursion={recursion} "
          f"gain={gain_mode} lict={findnet_lict} binarize={binarize}", flush=True)

    train_dir = os.path.join(args.data_dir, "train")
    meta_dir = os.path.join(args.data_dir, "meta")
    specs, metas = build_patient_specs(meta_dir, args.patient,
                                        min_z=args.min_z, max_z=args.max_z)
    if args.max_implants > 0:
        specs = specs[:args.max_implants]
    print(f"[debug] {args.patient}: {len(specs)} 序列 / {len(metas)} 样本", flush=True)

    loader = make_theta_sequence_loader(
        os.path.join(train_dir, "Target"), os.path.join(train_dir, "Baseline"),
        os.path.join(train_dir, "LI"), os.path.join(train_dir, "Mask"),
        os.path.join(train_dir, "Phys"))

    stats = {k: _new_st() for k in ["suff_tgt", "suff_pred", "glob_tgt", "glob_pred"]}
    picks = {"tissue": None, "transition": None, "artifact": None}

    with torch.no_grad():
        for i, spec in enumerate(specs):
            sample = loader(spec)
            thetas = [{k: v.unsqueeze(0).to(args.device) for k, v in t.items()}
                      for t in sample["thetas"]]
            x_hat = sample["x_hat"].unsqueeze(0).to(args.device)          # [1,Z,1,H,W]
            X_clean = sample["X_clean"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(args.device)
            metal_mask = sample["metal_mask"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(args.device)
            li = sample["li"].unsqueeze(0).to(args.device)
            Z = len(thetas)
            dirty_vol = x_hat.permute(0, 2, 1, 3, 4)                      # [1,1,Z,H,W]
            dirty = [dirty_vol[:, :, z] for z in range(Z)]
            li_vol = li.permute(0, 2, 1, 3, 4)
            li_list = [li_vol[:, :, z] for z in range(Z)]
            mask_input = (metal_mask if not config.binarize_mask
                          else (metal_mask > config.mask_bin_thr).float())
            masks = [mask_input[:, :, z] for z in range(Z)]
            observer = trainer._observer(dirty, masks, li_list)
            res = forward_backward(m_net, thetas, d_ref, recursion=recursion,
                                   observer=observer, gain_mode=config.gain_mode)
            x_hat_final = trainer._final_output(res, dirty, masks, li_list)  # [1,1,Z,H,W]

            I = 1.0 - metal_mask
            d_j = trainer._stack([t["d_j"] for t in thetas])               # [1,1,Z,H,W]
            r = d_j / d_ref
            suff = (I > 0.5) & (r > 9.0)
            nonmetal = I > 0.5
            tz = transition_zone_mask(metal_mask, r)                        # [1,1,Z,H,W]

            _acc(stats["suff_tgt"], X_clean[suff])
            _acc(stats["suff_pred"], x_hat_final[suff])
            _acc(stats["glob_tgt"], X_clean[nonmetal])
            _acc(stats["glob_pred"], x_hat_final[nonmetal])

            for z in range(Z):
                mf = float(metal_mask[0, 0, z].mean())
                sc = int(suff[0, 0, z].sum())
                tc = int(tz[0, 0, z].sum())
                snap = {"sid": spec[z],
                        "target": X_clean[0, 0, z].cpu().numpy(),
                        "baseline": dirty_vol[0, 0, z].cpu().numpy(),
                        "pred": x_hat_final[0, 0, z].cpu().numpy()}
                if mf < 0.005:
                    picks["tissue"] = _better(picks["tissue"], sc, snap)
                picks["transition"] = _better(picks["transition"], tc, snap)
                picks["artifact"] = _better(picks["artifact"], mf, snap)

            if (i + 1) % 25 == 0:
                print(f"  ... {i + 1}/{len(specs)} 序列", flush=True)

    print("\n===== 张量诊断（非金属，HU）=====", flush=True)
    for region, tag in [("充分区 r>9", "suff"), ("全局非金属", "glob")]:
        for sub in ["tgt", "pred"]:
            st = stats[f"{tag}_{sub}"]
            if st["n"] == 0:
                continue
            mean = st["sum"] / st["n"]
            var = max(0.0, st["ss"] / st["n"] - mean * mean)
            print(f"[{region}] {sub}: n={st['n']} min={st['mn']:.2f} max={st['mx']:.2f} "
                  f"mean={mean:.2f} std={np.sqrt(var):.2f}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    VWIN = (-150.0, 350.0)
    for name, pick in picks.items():
        if pick is None:
            print(f"[skip] {name}: no slice", flush=True)
            continue
        tgt, base, pred = pick["target"], pick["baseline"], pick["pred"]
        err = np.abs(pred - tgt)
        fig, axes = plt.subplots(1, 4, figsize=(21, 5.5))
        for ax, arr, tt in zip(axes[:3], [tgt, base, pred],
                               ["Target", "Baseline", "Prediction"]):
            ax.imshow(arr, cmap="gray", vmin=VWIN[0], vmax=VWIN[1],
                      interpolation="nearest")
            ax.set_title(tt, fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])
        im = axes[3].imshow(err, cmap="hot", vmin=0.0, vmax=350.0,
                            interpolation="nearest")
        axes[3].set_title("Error |pred-tgt|", fontsize=12)
        axes[3].set_xticks([]); axes[3].set_yticks([])
        fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
        fig.suptitle(f"{name}  sid={pick['sid']}  "
                     f"(pred min/max/mean={pred.min():.1f}/{pred.max():.1f}/{pred.mean():.1f} | "
                     f"tgt min/max/mean={tgt.min():.1f}/{tgt.max():.1f}/{tgt.mean():.1f})",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = os.path.join(args.out_dir, f"{name}.png")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[png] {out}", flush=True)

    print("[ALL DONE]", flush=True)


if __name__ == "__main__":
    main()
