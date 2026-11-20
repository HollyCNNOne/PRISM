#!/usr/bin/env python
"""stage1_findnet.py —— Stage 1：纯 2D FIND-Net 基线预训练（不含 SSM 递归），带验证+早停。

目标：用 Baseline + LI + Mask 强迫 FIND-Net 把放射状条纹擦干净（LIct=LI，即 FIND-Net
自己的经典基线）。所有 3D 序列打散成单切片，强增强（旋转 ±15°/弹性/Cutout），
损失 = masked MSE（(1−mask)·(x̂−Target)² / HU_SCALE²，FIND-Net Eq.6）。只训 FIND-Net，
输出权重供 Stage 2 暖启动，不进 SSM。

**验证 + 早停（2026-11-12 新增）**：
    - 验证集 = --val-patient（默认 patients_head2，零泄漏：训练与 d_ref 都排除它）。每 epoch
      末在验证集（无增广、no_grad）算 masked-MSE，与训练 loss 并列打印。
    - EarlyStopping：维护最近 --patience 个 epoch 的验证 loss 滑窗，若窗口内相对下降
      `(first−last)/first < --min-improvement` 即判定平台期，自动停止（不干等 200 epoch）。
    - 权重保存：验证 loss 新低 → 覆盖 --out（默认 findnet_stage1.pt，Stage 2 暖启动直接吃）；
      每 --save-every 个 epoch 另存 findnet_stage1_epoch{N}.pt 供 resume。
    - --resume：warm-start 已有权重（如 epoch20 的 findnet_stage1.pt），从 checkpoint 里的
      epoch 继续，不浪费已花的算力。

用法（服务器 pzneuro 环境）：
    /root/miniconda3/envs/pzneuro/bin/python -u scripts/stage1_findnet.py \
        --data-dir /root/autodl-tmp/PRISM/training_data_astra \
        --epochs 200 --batch-size 2 --lr 2e-4 --holdout-patient patients_head2 \
        --val-patient patients_head2 --patience 10 --device cuda
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import deque

import numpy as np
import torch
from torch.utils.data import DataLoader

_FRAMEWORK_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _FRAMEWORK_ROOT)

from prism.models.find_net import FINDNet, FINDNetArgs
from prism.models.find_net_gf import FINDNet as FINDNetGF, FINDNetArgs as FINDNetArgsGF
from prism.models.r_theta import HU_SCALE
from prism.data.dataset2d import Slice2DDataset, collate2d


def build_sample_ids(meta_dir, max_slices=0, holdout_patient=None, val_patient=None):
    """读 split=train 的样本；holdout_patient 排除、val_patient 只取（二者互斥用法）。"""
    ids = []
    for f in sorted(glob.glob(os.path.join(meta_dir, "*.json"))):
        m = json.load(open(f))
        if m.get("split", "train") != "train":
            continue
        if val_patient is not None:
            if m.get("patient_id") == val_patient:
                ids.append(m["sample_id"])
        else:
            if holdout_patient and m.get("patient_id") == holdout_patient:
                continue
            ids.append(m["sample_id"])
    if max_slices > 0 and val_patient is None:
        ids = ids[:max_slices]
    return ids


def masked_mse(x_hat, target, mask):
    """FIND-Net Eq.6：非金属区（1−mask）加权 MSE，残差按 HU_SCALE 归一化。"""
    resid = (x_hat - target) / HU_SCALE
    se = resid ** 2
    m = (1.0 - mask)
    return (se * m).sum() / (m.sum() + 1e-12)


def _evaluate(find_net, dl_val, device):
    """验证集 masked-MSE（无增广、no_grad），返回标量 float。"""
    find_net.eval()
    tot = 0.0
    n = 0
    with torch.no_grad():
        for ct_ma, lict, mask, target in dl_val:
            ct_ma = ct_ma.to(device)
            lict = lict.to(device)
            mask = mask.to(device)
            target = target.to(device)
            x_hat = find_net.reconstruct(ct_ma, lict, mask)
            tot += float(masked_mse(x_hat, target, mask).item()) * ct_ma.size(0)
            n += ct_ma.size(0)
    find_net.train()
    return tot / max(n, 1)


def _run_fullhead_stage1(find_net, rec, device):
    """在 held-out 完整头颅序列上逐片跑 FIND-Net（2D），返回 [Z,H,W] 输出（HU）。"""
    find_net.eval()
    outs = []
    with torch.no_grad():
        for z in range(rec.Z):
            ct_ma = torch.from_numpy(rec.dirty[z][None, None].astype(np.float32)).to(device)
            lict = torch.from_numpy(rec.li[z][None, None].astype(np.float32)).to(device)
            mask = torch.from_numpy(rec.metal[z][None, None].astype(np.float32)).to(device)
            x_hat = find_net.reconstruct(ct_ma, lict, mask)
            outs.append(x_hat[0, 0].cpu().numpy())
    find_net.train()
    return np.stack(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data_astra")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5,
                    help="v2.1：初始学习率降至 5e-5，防早停过早/震荡")
    ap.add_argument("--cosine-t-max", type=int, default=200,
                    help="v2.1：Cosine 退火周期")
    ap.add_argument("--cosine-eta-min", type=float, default=1e-6,
                    help="v2.1：Cosine 退火末值 lr")
    ap.add_argument("--no-early-stop", action="store_true",
                    help="v2.1：关闭早停，跑满 --epochs")
    ap.add_argument("--snapshot-epochs", type=str, default="50,100,150,200",
                    help="v2.1：网格快照 epoch（逗号分隔），存 data_dir/checkpoints/")
    ap.add_argument("--gf", action="store_true",
                    help="v2.2：用 GF 版 FIND-Net（FourierUnit 加可学习 Gaussian 带通）")
    ap.add_argument("--snapshot-prefix", type=str, default="stage1",
                    help="快照文件名前缀（GF 用 stage1_gf 避免覆盖 No-GF 快照）")
    ap.add_argument("--freq-record", action="store_true",
                    help="V2.2：每 epoch 记录 held-out 完整头颅的 3 频带 z 轴占比")
    ap.add_argument("--freq-out", type=str,
                    default="/root/autodl-tmp/PRISM/v2.2/fre_image/fre_epoch/freq_stage1.csv",
                    help="频域记录 CSV 输出路径")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-slices", type=int, default=0, help="0=全部；>0 取前 N 片 smoke")
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--holdout-patient", default=None,
                    help="零泄漏切分：预训练排除该病人（如 patients_head2，最终盲测）")
    ap.add_argument("--val-patient", default=None,
                    help="验证集病人（默认同 holdout=patients_head2；每 epoch 末算验证 loss）")
    ap.add_argument("--val-batch-size", type=int, default=4)
    ap.add_argument("--patience", type=int, default=10,
                    help="验证 loss 滑窗长度；窗口内相对下降 < --min-improvement 即早停")
    ap.add_argument("--min-improvement", type=float, default=0.01,
                    help="早停相对改进阈值（1%=0.01）：(first−last)/first < 此值判定平台期")
    ap.add_argument("--resume", default=None, help="warm-start 权重路径（含 config.epoch）")
    ap.add_argument("--out", default=None, help="最佳权重输出路径（默认 data_dir/findnet_stage1.pt）")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_dir = os.path.join(args.data_dir, "train")
    val_patient = args.val_patient or args.holdout_patient
    sample_ids = build_sample_ids(os.path.join(args.data_dir, "meta"), args.max_slices,
                                  args.holdout_patient)
    val_ids = build_sample_ids(os.path.join(args.data_dir, "meta"), 0,
                               val_patient=val_patient) if val_patient else []
    print(f"[data] train={len(sample_ids)} slices (2D flattened, augment={not args.no_augment}, "
          f"holdout={args.holdout_patient!r})  val={len(val_ids)} slices (patient={val_patient!r})",
          flush=True)

    ds = Slice2DDataset(
        sample_ids,
        os.path.join(train_dir, "Target"),
        os.path.join(train_dir, "Baseline"),
        os.path.join(train_dir, "LI"),
        os.path.join(train_dir, "Mask"),
        augment=not args.no_augment)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=collate2d, num_workers=args.num_workers, drop_last=True)

    dl_val = None
    if val_ids:
        ds_val = Slice2DDataset(val_ids,
                                os.path.join(train_dir, "Target"),
                                os.path.join(train_dir, "Baseline"),
                                os.path.join(train_dir, "LI"),
                                os.path.join(train_dir, "Mask"),
                                augment=False)
        dl_val = DataLoader(ds_val, batch_size=args.val_batch_size, shuffle=False,
                            collate_fn=collate2d, num_workers=args.num_workers)

    find_net = (FINDNetGF(FINDNetArgsGF()) if args.gf else FINDNet(FINDNetArgs())).to(args.device)
    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=args.device, weights_only=False)
        find_net.load_state_dict(ck["find_net"])
        start_epoch = ck.get("config", {}).get("epoch", 0)
        print(f"[load] resume find_net from {args.resume} (epoch {start_epoch})", flush=True)
    opt = torch.optim.Adam(find_net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.cosine_t_max, eta_min=args.cosine_eta_min)
    snapshot_epochs = {int(x) for x in args.snapshot_epochs.split(",") if x.strip()}
    ckpt_dir = os.path.join(args.data_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    n_params = sum(p.numel() for p in find_net.parameters())
    print(f"[model] FINDNet params={n_params} lr={args.lr} device={args.device} "
          f"cosine(T_max={args.cosine_t_max},eta_min={args.cosine_eta_min}) "
          f"early_stop={not args.no_early_stop} snapshots={sorted(snapshot_epochs)}", flush=True)

    out = args.out or os.path.join(args.data_dir, "findnet_stage1.pt")
    best_val = float("inf")
    val_hist = deque(maxlen=args.patience)

    # V2.2 频域记录：每个 epoch 记录 held-out 完整头颅的 3 频带 z 轴占比
    freq_rec = None
    if args.freq_record:
        from prism.scripts.freq_recorder import FreqRecorder
        freq_rec = FreqRecorder(args.data_dir,
                                patient=(val_patient or args.holdout_patient or "patients_head2"),
                                out_csv=args.freq_out)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tot = 0.0
        n = 0
        for ct_ma, lict, mask, target in dl:
            ct_ma = ct_ma.to(args.device)
            lict = lict.to(args.device)
            mask = mask.to(args.device)
            target = target.to(args.device)
            opt.zero_grad()
            x_hat = find_net.reconstruct(ct_ma, lict, mask)
            loss = masked_mse(x_hat, target, mask)
            if not torch.isfinite(loss):
                print(f"[NaN] epoch={epoch} loss={loss.item()}", flush=True)
                raise SystemExit(1)
            loss.backward()
            opt.step()
            tot += float(loss.item())
            n += 1
        train_loss = tot / max(n, 1)
        dt = time.time() - t0

        # 验证集 loss（无增广、no_grad）
        val_loss = _evaluate(find_net, dl_val, args.device) if dl_val is not None else float("nan")

        # V2.2 频域记录
        if freq_rec is not None:
            try:
                out_vol = _run_fullhead_stage1(find_net, freq_rec, args.device)
                freq_rec.record(out_vol, epoch)
            except Exception as e:
                print(f"[freq] 记录失败 epoch={epoch}: {e}", flush=True)

        # 最佳权重（验证 loss 新低 → 覆盖 --out）
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            torch.save({"find_net": find_net.state_dict(),
                        "config": {"epoch": epoch + 1, "lr": args.lr, "best_val": best_val}}, out)
            flag = " *BEST*"
        else:
            flag = ""

        # 早停判定：最近 patience 个 epoch 验证 loss 相对下降 < min_improvement（v2.1 可 --no-early-stop 关闭）
        val_hist.append(val_loss)
        stop = False
        if not args.no_early_stop:
            if len(val_hist) == args.patience and val_hist[0] > 0:
                rel_improv = (val_hist[0] - val_hist[-1]) / val_hist[0]
                if rel_improv < args.min_improvement:
                    stop = True

        vstr = f"val={val_loss:.4f}" if val_loss == val_loss else "val=nan"
        bstr = f"best={best_val:.4f}" if best_val != float("inf") else "best=inf"
        print(f"[epoch {epoch + 1}/{args.epochs}] train={train_loss:.4f} {vstr} {bstr} "
              f"batches={n} {dt:.1f}s{flag}", flush=True)

        # v2.1：网格快照（epoch 50/100/150/200 强制保存到 checkpoints/）
        if (epoch + 1) in snapshot_epochs:
            snap = os.path.join(ckpt_dir, f"{args.snapshot_prefix}_epoch{epoch + 1}.pt")
            torch.save({"find_net": find_net.state_dict(),
                        "config": {"epoch": epoch + 1, "lr": args.lr, "best_val": best_val}}, snap)
            print(f"[snapshot] {snap}", flush=True)

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save({"find_net": find_net.state_dict(),
                        "config": {"epoch": epoch + 1, "lr": args.lr, "best_val": best_val}},
                       os.path.join(args.data_dir, f"findnet_stage1_epoch{epoch + 1}.pt"))

        scheduler.step()

        if stop:
            print(f"[early-stop] 验证 loss 最近 {args.patience} epoch 相对下降 < "
                  f"{args.min_improvement}，平台期，停止于 epoch {epoch + 1}。最佳 val={best_val:.4f} "
                  f"已存 {out}", flush=True)
            break

    if best_val != float("inf"):
        print(f"[done] 最佳验证 loss={best_val:.4f} → {out}", flush=True)
    else:
        torch.save({"find_net": find_net.state_dict(),
                    "config": {"epoch": args.epochs, "lr": args.lr}}, out)
        print(f"[done] 无验证集，末轮权重 → {out}", flush=True)

    if freq_rec is not None:
        csv_path = freq_rec.save()
        print(f"[freq] 频域记录已存 {csv_path}", flush=True)


if __name__ == "__main__":
    main()
