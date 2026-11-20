"""verify_dataset.py — 数据契约三关验证（不过即 SystemExit，禁止进入训练）。

关 1  d_j 尺度：训练集全像素 d_j 中位数应在 1e5~1e8（平行束解析线 ≈4.1e6）；
                 1e9 量级 = 扇束残留 → 失败。
关 2  域一致：取一张**无金属**切片，v2.1 FIND-Net 输出相对 GT 的偏置应 |bias| < 10 HU
                 （扇束数据上会到 +50 以上）。
关 3  分区比例：r=d_j/d_ref，完全(r<1/9)/部分/充分(r>9) 应大致复现 21.6 / 74.5 / 3.9%。
"""
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
    net = FINDNet(FINDNetArgs()).to(args.device)
    net.load_state_dict(torch.load(args.findnet, map_location=args.device,
                                   weights_only=False)["find_net"])
    net.eval()
    # 找一张无金属切片
    clean_id = None
    for f in sorted(glob.glob(os.path.join(args.data_dir, "train", "Mask", "*.raw")))[::11]:
        sid = os.path.basename(f)[:-4]
        if read_raw(f).max() < 1e-3:
            clean_id = sid
            break
    if clean_id is None:
        fails.append("关2 找不到无金属切片")
        print("[关2] 找不到无金属切片 → FAIL")
    else:
        gt = read_raw(os.path.join(args.data_dir, "train", "Target", f"{clean_id}.raw"))
        li = read_raw(os.path.join(args.data_dir, "train", "LI", f"{clean_id}.raw"))
        z = np.zeros_like(gt)
        t = lambda a: torch.from_numpy(a[None, None].astype(np.float32)).to(args.device)
        with torch.no_grad():
            out = net.reconstruct(t(gt), t(li), t(z))[0, 0].cpu().numpy()
        tis = gt > -500.0
        bias = float((out - gt)[tis].mean())
        ok2 = abs(bias) < 10.0
        print(f"[关2] 干净图 FIND-Net 偏置 = {bias:+.1f} HU（|bias|<10）→ {'PASS' if ok2 else 'FAIL'}  [{clean_id}]")
        if not ok2:
            fails.append(f"关2 域不一致（偏置 {bias:+.1f} HU）")

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
        vals.append(dj[mk <= 1e-3])
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
