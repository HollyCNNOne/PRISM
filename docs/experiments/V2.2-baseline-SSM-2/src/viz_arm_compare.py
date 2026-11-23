"""viz_arm_compare.py — 两臂输出肉眼对比（held-out 代表片）。

渲染：GT / Baseline / 臂1 / 臂2（上排），对应误差图（下排），另出金属局部放大。
用法：
    python viz_arm_compare.py --ckpts a.pt,b.pt --names baseline,exp008 --out xxx.png
"""
import argparse, glob, json, os, sys
import numpy as np
import torch
from scipy.ndimage import maximum_filter

_FR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _FR)
from prism.data.dataset import make_theta_sequence_loader
from prism.models.m_network import MNetwork
from prism.models.find_net import FINDNet, FINDNetArgs
from prism.recursion.algorithm1 import forward_backward
from prism.training.trainer import SSMTrainer
from prism.utils.config import SSMConfig

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MASK_EPS = 1e-3


def read_raw(p):
    return np.fromfile(p, np.float32).reshape(512, 512)


def find_split(d, sid, sub):
    for sp in ("train", "val"):
        p = os.path.join(d, sp, sub, f"{sid}.raw")
        if os.path.exists(p):
            return p
    return None


def build(ckpt, dev):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = ck["config"]
    m = MNetwork(nb=cfg.get("nb", 1), channels=cfg.get("channels", 32),
                 use_velocity=cfg.get("use_velocity", True),
                 shadow_r0=cfg.get("shadow_r0", 1.0 / 9.0),
                 use_z_curvature=cfg.get("use_z_curvature", False)).to(dev)
    fn = FINDNet(FINDNetArgs()).to(dev)
    m.load_state_dict(ck["model"]); fn.load_state_dict(ck["find_net"])
    c = SSMConfig(nb=cfg.get("nb", 1), channels=cfg.get("channels", 32),
                  d_ref=cfg["d_ref"], gain_mode=cfg.get("gain_mode", "kalman"),
                  findnet_lict_source=cfg.get("findnet_lict_source", "mu"),
                  backfill_r=cfg.get("backfill_r", 8),
                  recursion=cfg.get("recursion", "bidirectional"),
                  shadow_r0=cfg.get("shadow_r0", 1.0 / 9.0),
                  zfreq_gain=cfg.get("zfreq_gain", 0.0))
    return m, fn, c, SSMTrainer(m, c, device=dev, find_net=fn)


def run(m, fn, cfg, trainer, data_dir, sids, dev):
    loader = make_theta_sequence_loader(
        os.path.join(data_dir, "train", "Target"), os.path.join(data_dir, "train", "Baseline"),
        os.path.join(data_dir, "train", "LI"), os.path.join(data_dir, "train", "Mask"),
        os.path.join(data_dir, "train", "Phys"))
    sm = loader(sids); Z = len(sids)
    th = [{k: v.unsqueeze(0).to(dev) for k, v in t.items()} for t in sm["thetas"]]
    mm = sm["metal_mask"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(dev)
    xh = sm["x_hat"].unsqueeze(0).to(dev); li = sm["li"].unsqueeze(0).to(dev)
    dv = xh.permute(0, 2, 1, 3, 4); lv = li.permute(0, 2, 1, 3, 4)
    dirty = [dv[:, :, z] for z in range(Z)]
    li_l = [lv[:, :, z] for z in range(Z)]
    masks = [mm[:, :, z] for z in range(Z)]
    obs = trainer._observer(dirty, masks, li_l)
    m.eval(); fn.eval()
    with torch.no_grad():
        res = forward_backward(m, th, cfg.d_ref, recursion=cfg.recursion,
                               observer=obs, gain_mode=cfg.gain_mode)
        out = trainer._final_output(res, dirty, masks, li_l)
    m.train(); fn.train()
    return out[0, 0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data_astra")
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--names", default=None)
    ap.add_argument("--patient", default="patients_head2")
    ap.add_argument("--n-slices", type=int, default=3)
    ap.add_argument("--out", default="/root/autodl-tmp/PRISM/v2.2/arm_compare_exp008/viz.png")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ckpts = [c.strip() for c in args.ckpts.split(",") if c.strip()]
    names = (args.names.split(",") if args.names else
             [os.path.basename(c).replace(".pt", "") for c in ckpts])
    dev = args.device

    meta = {json.load(open(f))["sample_id"]: json.load(open(f))
            for f in glob.glob(os.path.join(args.data_dir, "meta", "*.json"))}
    by_seq = {}
    for sid, m in meta.items():
        if m.get("patient_id") == args.patient:
            by_seq.setdefault(m["sequence_id"], []).append(m)

    # 每条序列取金属量最大的片；再按金属量取前 N 大的序列
    rep = []
    for seq, ms in by_seq.items():
        best, bv = None, -1
        for m in ms:
            p = find_split(args.data_dir, m["sample_id"], "Mask")
            if not p:
                continue
            mk_ = read_raw(p)
            v = float((mk_ > MASK_EPS).sum())
            if v > bv:
                best, bv = (ms, m["sample_id"]), v
        if best:
            rep.append((bv, best[0], best[1]))
    rep.sort(key=lambda x: -x[0])
    rep = rep[:args.n_slices]
    print("[pick]", [(r[2], int(r[0])) for r in rep], flush=True)

    arms = [build(c, dev) for c in ckpts]
    panels = []
    for mval, sids, pick in rep:
        sid_list = sorted(sids, key=lambda x: int(x["sample_id"].rsplit("_", 1)[1]))
        ids = [m["sample_id"] for m in sid_list]
        zi = ids.index(pick)
        gt = read_raw(find_split(args.data_dir, pick, "Target"))
        bl = read_raw(find_split(args.data_dir, pick, "Baseline"))
        mk = read_raw(find_split(args.data_dir, pick, "Mask"))
        outs = []
        for (mo, fo, co, to) in arms:
            o = run(mo, fo, co, to, args.data_dir, ids, dev)
            outs.append(o[zi])
        panels.append((pick, mval, gt, bl, mk, outs))
        print("[done]", pick, flush=True)

    n = len(ckpts); P = len(panels)
    fig, axes = plt.subplots(2 * P, 1 + n, figsize=(3.7 * (1 + n), 3.6 * 2 * P))
    if 2 * P == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)
    imsh = None
    for r, (pick, mval, gt, bl, mk, outs) in enumerate(panels):
        imgs = [("GT", gt)] + [(nm, o) for nm, o in zip(names, outs)]
        for c, (nm, im) in enumerate(imgs):
            ax = axes[2 * r, c]
            ax.imshow(im, cmap="gray", vmin=-200, vmax=600)
            title = nm if c != 0 else (nm + "  |  " + pick + "  metal=" + f"{mval:.0f}")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            ax2 = axes[2 * r + 1, c]
            if c == 0:
                ax2.axis("off")
                ax2.text(.5, .5, "|·−GT|", ha="center", va="center",
                         transform=ax2.transAxes, fontsize=11)
            else:
                e = np.abs(im - gt)
                imsh = ax2.imshow(e, cmap="inferno", vmin=0, vmax=400)
                ax2.set_title(f"|{nm}−GT|", fontsize=9); ax2.axis("off")
    if imsh is not None:
        fig.colorbar(imsh, ax=axes[:, 1:], fraction=0.015, pad=0.01, label="HU")
    fig.suptitle("Arm comparison on held-out representative slices", fontsize=12)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[save]", args.out)


if __name__ == "__main__":
    main()
