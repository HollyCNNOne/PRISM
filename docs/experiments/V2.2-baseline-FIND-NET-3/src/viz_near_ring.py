"""viz_near_ring.py — exp014 的区域示意图：近环（金属外 8–30px）在图上长什么样。

输出两行四列（每个面板叠加近环轮廓）：
    上排：GT / Baseline / FIND-Net（纯 2D） / FIND-Net+SSM
    下排：对应误差 |· − GT|（同一色标）

另出一张局部放大（金属中心裁剪），让 3.5 倍假梯度肉眼可见。
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
from matplotlib.patches import Rectangle

MASK_EPS = 1e-3
NEAR_IN, NEAR_OUT = 8, 30


def read_raw(p):
    return np.fromfile(p, np.float32).reshape(512, 512)


def find_split(d, sid, sub):
    for sp in ("train", "val"):
        p = os.path.join(d, sp, sub, f"{sid}.raw")
        if os.path.exists(p):
            return p
    return None


def dilate2d(b, r):
    return maximum_filter(b, size=(2 * r + 1, 2 * r + 1), mode="constant", cval=False)


def run(m_net, find_net, config, trainer, data_dir, sids, dev):
    loader = make_theta_sequence_loader(
        os.path.join(data_dir, "train", "Target"), os.path.join(data_dir, "train", "Baseline"),
        os.path.join(data_dir, "train", "LI"), os.path.join(data_dir, "train", "Mask"),
        os.path.join(data_dir, "train", "Phys"))
    sm = loader(sids)
    Z = len(sids)
    th = [{k: v.unsqueeze(0).to(dev) for k, v in t.items()} for t in sm["thetas"]]
    mm = sm["metal_mask"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(dev)
    xh = sm["x_hat"].unsqueeze(0).to(dev)
    li = sm["li"].unsqueeze(0).to(dev)
    dv = xh.permute(0, 2, 1, 3, 4); lv = li.permute(0, 2, 1, 3, 4)
    dirty = [dv[:, :, z] for z in range(Z)]
    li_l = [lv[:, :, z] for z in range(Z)]
    masks = [mm[:, :, z] for z in range(Z)]
    obs = trainer._observer(dirty, masks, li_l)
    m_net.eval(); find_net.eval()
    with torch.no_grad():
        res = forward_backward(m_net, th, config.d_ref, recursion=config.recursion,
                               observer=obs, gain_mode=config.gain_mode)
        out = trainer._final_output(res, dirty, masks, li_l)
        fn = torch.stack([obs(dirty[z], z) for z in range(Z)], dim=2)
    m_net.train(); find_net.train()
    return out[0, 0].cpu().numpy(), fn[0, 0, :Z].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data_astra")
    ap.add_argument("--ckpt", default="/root/autodl-tmp/PRISM/training_data_astra/ssm_stage2_v22.pt")
    ap.add_argument("--patient", default="patients_head2")
    ap.add_argument("--out", default="/root/autodl-tmp/PRISM/v2.2/clinical/near_ring_viz.png")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = ck["config"]
    m_net = MNetwork(nb=cfg.get("nb", 1), channels=cfg.get("channels", 32),
                     use_velocity=cfg.get("use_velocity", True),
                     shadow_r0=cfg.get("shadow_r0", 1.0 / 9.0),
                     use_z_curvature=cfg.get("use_z_curvature", False)).to(dev)
    find_net = FINDNet(FINDNetArgs()).to(dev)
    m_net.load_state_dict(ck["model"]); find_net.load_state_dict(ck["find_net"])
    config = SSMConfig(nb=cfg.get("nb", 1), channels=cfg.get("channels", 32),
                       d_ref=cfg["d_ref"], gain_mode=cfg.get("gain_mode", "kalman"),
                       findnet_lict_source=cfg.get("findnet_lict_source", "mu"),
                       backfill_r=cfg.get("backfill_r", 8),
                       recursion=cfg.get("recursion", "bidirectional"),
                       shadow_r0=cfg.get("shadow_r0", 1.0 / 9.0))
    trainer = SSMTrainer(m_net, config, device=dev, find_net=find_net)

    meta = {json.load(open(f))["sample_id"]: json.load(open(f))
            for f in glob.glob(os.path.join(args.data_dir, "meta", "*.json"))}
    by_seq = {}
    for sid, m in meta.items():
        if m.get("patient_id") == args.patient:
            by_seq.setdefault(m["sequence_id"], []).append(sid)
    specs = [sorted(v, key=lambda x: int(x.rsplit("_", 1)[1])) for v in by_seq.values()]
    specs = [s for s in specs if len(s) >= 60]
    specs.sort(key=lambda s: -len(s))

    # 选金属最多的一片所在的序列
    best = None
    for sp in specs[:3]:
        for sid in sp:
            mk = read_raw(find_split(args.data_dir, sid, "Mask"))
            v = float((mk > MASK_EPS).sum())
            if best is None or v > best[0]:
                best = (v, sp, sid)
    _, spec, pick = best
    print(f"[pick] seq={spec[0]} slice={pick} metal_vox={best[0]:.0f}", flush=True)

    gt_all = np.stack([read_raw(find_split(args.data_dir, s, "Target")) for s in spec])
    mk_all = np.stack([read_raw(find_split(args.data_dir, s, "Mask")) for s in spec])
    bl_all = np.stack([read_raw(find_split(args.data_dir, s, "Baseline")) for s in spec])
    out, fn = run(m_net, find_net, config, trainer, args.data_dir, spec, dev)

    zi = spec.index(pick)
    gt, mk, bl = gt_all[zi], mk_all[zi], bl_all[zi]
    o, f = out[zi], fn[zi]
    core = mk > MASK_EPS
    near = dilate2d(core, NEAR_OUT) & (~dilate2d(core, NEAR_IN))
    ys, xs = np.where(core)
    cy, cx = int(ys.mean()), int(xs.mean())

    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                         "font.size": 9})
    panels = [("GT", gt, None), ("Baseline", bl, None),
              ("FIND-Net (2D)", f, None), ("FIND-Net+SSM", o, None)]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6))
    vmin, vmax = -200, 600
    for j, (name, img, _) in enumerate(panels):
        ax = axes[0, j]
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.contour(near, levels=[0.5], colors=["#00e5ff"], linewidths=0.7)
        ax.contour(core, levels=[0.5], colors=["#ff3b30"], linewidths=0.8)
        ax.set_title(name); ax.axis("off")

        ax = axes[1, j]
        if j == 0:
            ax.axis("off"); ax.text(.5, .5, "error maps\n|· − GT|", ha="center", va="center",
                                    fontsize=11, transform=ax.transAxes)
        else:
            e = np.abs(img - gt)
            im = ax.imshow(e, cmap="inferno", vmin=0, vmax=400)
            ax.contour(near, levels=[0.5], colors=["#00e5ff"], linewidths=0.7)
            ax.set_title(f"|{name} − GT|"); ax.axis("off")
    fig.colorbar(im, ax=axes[1, :], fraction=0.02, pad=0.01, label="HU")
    fig.suptitle("Near ring (metal +8..30 px): cyan = near-ring boundary, red = metal core", fontsize=11)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[save]", args.out)

    # ---- 局部放大 ----
    h = 110
    y0, y1 = max(0, cy - h), min(512, cy + h)
    x0, x1 = max(0, cx - h), min(512, cx + h)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    for j, (name, img, _) in enumerate(panels):
        ax = axes[j]
        ax.imshow(img[y0:y1, x0:x1], cmap="gray", vmin=vmin, vmax=vmax)
        ax.contour(near[y0:y1, x0:x1], levels=[0.5], colors=["#00e5ff"], linewidths=0.8)
        ax.contour(core[y0:y1, x0:x1], levels=[0.5], colors=["#ff3b30"], linewidths=0.9)
        ax.set_title(name); ax.axis("off")
    fig.suptitle("Zoom: the near ring is where fake gradients appear (FIND-Net 3.48x)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    zoom = os.path.join(os.path.dirname(args.out), "near_ring_zoom.png")
    fig.savefig(zoom, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[save]", zoom)


if __name__ == "__main__":
    main()
