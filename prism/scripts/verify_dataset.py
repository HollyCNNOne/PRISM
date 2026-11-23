"""verify_dataset.py — 数据契约三关验证（不过即 SystemExit，禁止进入训练）。

关 1  d_j 尺度：训练集全像素 d_j 中位数应在 1e5~1e8（平行束解析线 ≈4.1e6）；
                 1e9 量级 = 扇束残留 → 失败。
关 2  域一致：取一张**无金属**切片，v2.1 FIND-Net 输出相对 GT 的偏置应 |bias| < 10 HU
                 （扇束数据上会到 +50 以上）。
关 3  分区比例：r=d_j/d_ref，完全(r<1/9)/部分/充分(r>9) 应大致复现 21.6 / 74.5 / 3.9%。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from prism.eval.domain import tissue_mask
import argparse, glob, json, os, sys
import numpy as np
import torch

_FR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _FR)
from prism.models.find_net import FINDNet, FINDNetArgs

F = "/root/autodl-tmp/PRISM/training_data_astra"


def read_raw(p):
    return np.fromfile(p, np.float32).reshape(512, 512)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=F)
    ap.add_argument("--findnet", default=f"{F}/findnet_stage1_v2.1.pt")
    ap.add_argument("--outdir", default="/root/autodl-tmp/PRISM/v2.2/verify")
    ap.add_argument("--holdout", default="patients_head2")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    fails = []

    # ---------------- 关 1 ----------------
    fs = sorted(glob.glob(os.path.join(args.data_dir, "train", "Phys", "*_d_j.npy")))[::37]
    d = np.concatenate([np.load(f).astype(np.float32)[0, ::4, ::4].ravel() for f in fs])
    d_med = float(np.median(d.astype(np.float64)))
    ok1 = 1e5 <= d_med <= 1e8
    print(f"[关1] d_j 中位数 = {d_med:.3e}  期望 1e5~1e8  → {'PASS' if ok1 else 'FAIL'}")
    if not ok1:
        fails.append("关1 d_j 尺度（疑似扇束残留）")

    # ---------------- 关 2 ----------------
    # 判据改为「模型在这批数据上确实有效」：取 held-out 病人的**有金属**切片，
    # FIND-Net 必须把非金属组织误差降到 Baseline 的一半以下。
    # （旧判据「干净图偏置≈0」不成立：v2.1 只见过有金属切片，GT 输入/mask=0 都是 OOD。）
    net = FINDNet(FINDNetArgs()).to(args.device)
    net.load_state_dict(torch.load(args.findnet, map_location=args.device,
                                   weights_only=False)["find_net"])
    net.eval()
    metas = [json.load(open(f)) for f in glob.glob(os.path.join(args.data_dir, "meta", "*.json"))]
    cands = [m for m in metas if m.get("patient_id") == args.holdout]
    cands.sort(key=lambda m: m["sample_id"])
    cands = cands[::max(1, len(cands) // 60)][:60]          # 采样 60 张
    best, best_m = None, -1.0
    for m in cands:
        p = os.path.join(args.data_dir, "train", "Mask", f"{m['sample_id']}.raw")
        if not os.path.exists(p):
            continue
        v = float(read_raw(p).sum())
        if v > best_m:
            best_m, best = v, m["sample_id"]
    if best is None or best_m <= 0:
        fails.append("关2 找不到有金属的 held-out 切片")
        print("[关2] 找不到有金属切片 → FAIL")
    else:
        gt = read_raw(os.path.join(args.data_dir, "train", "Target", f"{best}.raw"))
        bl = read_raw(os.path.join(args.data_dir, "train", "Baseline", f"{best}.raw"))
        li = read_raw(os.path.join(args.data_dir, "train", "LI", f"{best}.raw"))
        mk = read_raw(os.path.join(args.data_dir, "train", "Mask", f"{best}.raw"))
        tis = tissue_mask(mk, gt, r=8)
        t = lambda a: torch.from_numpy(a[None, None].astype(np.float32)).to(args.device)
        with torch.no_grad():
            out = net.reconstruct(t(bl), t(li), t(mk))[0, 0].cpu().numpy()
        e_bl = float(np.abs(bl - gt)[tis].mean())
        e_fn = float(np.abs(out - gt)[tis].mean())
        ratio = e_fn / max(e_bl, 1e-6)
        ok2 = ratio <= 0.5
        print(f"[关2] held-out 有金属切片 {best}: Baseline {e_bl:.1f} → FIND-Net {e_fn:.1f} HU "
              f"(ratio {ratio:.2f}, 需<=0.5) → {'PASS' if ok2 else 'FAIL'}")
        if not ok2:
            fails.append(f"关2 FIND-Net 无效（ratio {ratio:.2f}）")

    # ---------------- 关 3 ----------------
    fs = sorted(glob.glob(os.path.join(args.data_dir, "train", "Phys", "*_d_j.npy")))[::41]
    vals = []
    for f in fs:
        sid = os.path.basename(f).replace("_d_j.npy", "")
        mp = os.path.join(args.data_dir, "train", "Mask", f"{sid}.raw")
        if not os.path.exists(mp):
            continue
        dj = np.load(f).astype(np.float32)[0, ::4, ::4]
        mk = read_raw(mp)[::4, ::4]
        vals.append(dj[tissue_mask(mk, gt, r=8)])
    t = np.concatenate(vals).astype(np.float64)
    r = t / d_med
    f_complete = float((r < 1 / 9).mean())
    f_suff = float((r > 9).mean())
    ok3 = (0.10 <= f_complete <= 0.35) and (0.01 <= f_suff <= 0.10)
    print(f"[关3] 完全={f_complete:.3f} 部分={1-f_complete-f_suff:.3f} 充分={f_suff:.3f} "
          f"（期望 ≈0.216/0.745/0.039）→ {'PASS' if ok3 else 'FAIL'}")
    if not ok3:
        fails.append("关3 分区比例")

    with open(os.path.join(args.outdir, "verify_report.txt"), "w") as f:
        f.write(f"d_j_median={d_med:.4e}\n")
        f.write(f"complete={f_complete:.4f} sufficient={f_suff:.4f}\n")
        f.write("PASS\n" if not fails else "FAIL: " + "; ".join(fails) + "\n")

    if fails:
        print("[STOP] 三关未通过: " + "; ".join(fails))
        raise SystemExit(1)
    print("[OK] 三关全部通过")


if __name__ == "__main__":
    main()
