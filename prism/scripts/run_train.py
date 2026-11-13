#!/usr/bin/env python
"""run_train.py —— PRISM SSM 训练入口（组织流形 M 的深度高斯状态空间模型）。

把 3D_Coil 产出的 2535 样本（286 植入体 / 4 病人）组进 Theta 序列，冻结 d_ref，
用 SSMTrainer 训练（Q 监督 + 物理一致性 L_pc + 下游 FIND-Net 去除损失 L_down）。

数据契约（见 data/dataset.py）：
    train/{Target,Baseline,LI,Mask}/*.raw            — 干净 / 伪影 / LI / 金属掩膜（512² f32 HU）
    train/Phys/{sample_id}_{x_M0,d_j,g,tau_b,w_b}.npy — 物理量
    meta/{sample_id}.json                             — {sample_id, sequence_id, split, ...}

分带说明：tau_b/w_b 当前是 0 占位（径向 LFR 未派生），本脚本默认 nb=1 标量训练；
分带（nb=4）等 AGIR 侧派生真实 tau_b/w_b 后再启用（make_theta_sequence_loader 会自动
在非零时注入分带量，本脚本无需改）。

用法：
    python run_train.py --data-dir /root/autodl-tmp/PRISM/training_data \
        --nb 1 --epochs 20 --batch-size 1 --device cuda --max-implants 0
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

# 本脚本位于 PRISM/scripts/ → 上溯两级到 prism_framework/（prism 包所在），加进 sys.path。
_FRAMEWORK_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _FRAMEWORK_ROOT)

from prism.data.dataset import ThetaDataset, collate_theta, make_theta_sequence_loader  # noqa: E402
from prism.models.m_network import MNetwork  # noqa: E402
from prism.training.trainer import SSMTrainer  # noqa: E402
from prism.utils.config import SSMConfig  # noqa: E402


def _sample_idx(sample_id: str) -> int:
    """sample_000129 → 129。z 序 = 数值序（3D_Coil 连续编号）。"""
    return int(sample_id.rsplit("_", 1)[1])


def build_specs(meta_dir: str, split: str = "train", holdout_patient: str | None = None,
                case_categories: list[str] | None = None):
    """扫 meta/*.json，按 sequence_id 分组 → (specs, metas)。

    specs: list[list[sample_id]]（每体 = 一次植入的 z 升序切片），供 Dataset 用；
    metas:  全部 split 样本的 meta list，供 d_ref 冻结用（全训练集，非 smoke 子集）。
    holdout_patient: 非 None 时按 patient_id 排除该病人（held-out 评估，防泄漏）。
    case_categories: 非 None 时只保留 case_category 在其中的样本（流水线开发：
        生成端按 case 分类先产出 complete，训练端同步只吃 complete，不必等全量）。
    """
    allowed = None if case_categories is None else set(case_categories)
    metas = []
    for f in glob.glob(os.path.join(meta_dir, "*.json")):
        m = json.load(open(f))
        if m.get("split", "train") != split:
            continue
        if holdout_patient and m.get("patient_id") == holdout_patient:
            continue
        if allowed is not None and m.get("case_category") not in allowed:
            continue
        metas.append(m)
    by_seq = defaultdict(list)
    for m in metas:
        by_seq[m["sequence_id"]].append(m["sample_id"])
    specs = [sorted(sids, key=_sample_idx) for sids in by_seq.values()]
    specs.sort(key=lambda s: (_sample_idx(s[0]), len(s)))
    return specs, metas


def freeze_d_ref(metas, phys_dir: str, cache_path: str, stride: int = 4) -> float:
    """从训练集全部 d_j 场采样冻结 d_ref = median_j d_j（C5），缓存到 cache_path。

    stride=4 → 每个 d_j 场取 1/16 体素，2535 场 ≈ 41.5M 值（float32 ≈ 166MB）。
    """
    if os.path.exists(cache_path):
        d_ref = float(np.load(cache_path)[0])
        print(f"[d_ref] 载入缓存 {cache_path}: {d_ref:.6g}", flush=True)
        return d_ref
    chunks = []
    for m in metas:
        a = np.load(os.path.join(phys_dir, f"{m['sample_id']}_d_j.npy"))  # [1,H,W]
        chunks.append(a[0, ::stride, ::stride].ravel())
    flat = np.concatenate(chunks).astype(np.float64)
    d_ref = float(np.median(flat))
    np.save(cache_path, np.array([d_ref], dtype=np.float64))
    print(f"[d_ref] 冻结 d_ref = {d_ref:.6g} (n_voxels={flat.size})", flush=True)
    return d_ref


def to_device(batch, device):
    thetas = [{k: v.to(device) for k, v in t.items()} for t in batch["thetas"]]
    return {
        "thetas": thetas,
        "X_clean": batch["X_clean"].to(device),
        "metal_mask": batch["metal_mask"].to(device),
        "x_hat": batch["x_hat"].to(device),
        "meta": batch["meta"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data")
    ap.add_argument("--nb", type=int, default=1,
                    help="分带数（当前数据 tau_b/w_b=0 → 只支持 nb=1）")
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--findnet-lr", type=float, default=2e-4,
                    help="FIND-Net 学习率（官方 Adam lr=2e-4，与 M_μ 解耦）")
    ap.add_argument("--max-implants", type=int, default=0,
                    help=">0 时只取前 N 植入体（smoke）")
    ap.add_argument("--min-z", type=int, default=3,
                    help="丢弃 Z < min_z 的过短序列（Q 窗口 kT=3 需要 Z>=3）")
    ap.add_argument("--max-z", type=int, default=0,
                    help=">0 时把序列截断到前 max_z 片（BPTT 显存上限，Z 过长会 OOM）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lambda-q", type=float, default=1.0)
    ap.add_argument("--lambda-pc", type=float, default=0.1)
    ap.add_argument("--lambda-down", type=float, default=1.0,
                    help="下游 FIND-Net 去除损失 L_down 权重（残差已按 100HU 归一化）")
    ap.add_argument("--lambda-mu", type=float, default=0.0,
                    help="组织先验 μ 直接保真损失 L_mu 权重（默认 0=关；>0 逼 μ≈X_clean，"
                         "消除 x̂_final=μ+corr 退化，让递归真正 carry 组织进阴影区）")
    ap.add_argument("--trust-mode", default="kalman", choices=["kalman", "gated", "fixed"],
                    help="C 消融（H2）信任信号机制（已废弃，仅旧诊断脚本用）：kalman=后验总精度 "
                         "p_post_sum / gated=旧 π=g⊙c（gate.py 复原）/ fixed=learnable 标量 γ")
    ap.add_argument("--recursion", default="bidirectional",
                    choices=["bidirectional", "forward", "per_slice"],
                    help="B 消融（H3）递归模式：bidirectional=双向融合（现状）/ "
                         "forward=单向 / per_slice=无递归基线")
    ap.add_argument("--no-velocity", action="store_true",
                    help="B 消融（H3）B1 去速度：速度项 Δμ 不进 backbone/M_μ（置 0），"
                         "纯位置模型")
    ap.add_argument("--gate-reads-dj", action="store_true",
                    help="根因修复#2：门控 Δ 读 d_j/d_ref（骨干输入 2ch→3ch），让 Δ 能看见"
                         "阴影、学 Δ→0（carry 上一片干净先验）")
    ap.add_argument("--no-xm0", action="store_true",
                    help="A 消融（H1）桥消融：M_μ 转移算子输入去掉物理量 x_M0（置 0）")
    ap.add_argument("--no-dj", action="store_true",
                    help="A 消融（H1）桥消融：M_μ 输入去掉物理量 d_j（置 0；阴影门/增益的 d_j 保留）")
    ap.add_argument("--no-g", action="store_true",
                    help="A 消融（H1）桥消融：M_μ 输入去掉物理量 g（置 0）")
    ap.add_argument("--holdout-patient", default=None,
                    help="留出评估的病人 patient_id（不参与训练，按 patient_id 过滤 meta；"
                         "None=全量训练）。held-out 协议：训练 3 病人、评 1 病人。")
    ap.add_argument("--case-categories", default=None,
                    help="流水线开发：只训练这些 case_category 的样本（逗号分隔，如 "
                         "complete,large_residue；None=全量）。与生成端 --case_categories "
                         "配对：生成端先出 complete，训练端先吃 complete。")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--clip-grad", type=float, default=0.0,
                    help=">0 时对总损失做 grad-norm 裁剪")
    ap.add_argument("--lr-schedule", default="none", choices=["none", "cosine"],
                    help="LR 调度：none=恒定（现状）/ cosine=余弦退火到 ~0")
    ap.add_argument("--warmup-epochs", type=int, default=0,
                    help="cosine 调度下的线性 warmup epoch 数（0=无 warmup）")
    ap.add_argument("--save-every", type=int, default=0,
                    help=">0 时每 N epoch 存一次 checkpoint（长训练防丢）")
    ap.add_argument("--tag", default="",
                    help="checkpoint 名附加后缀（区分长训练 run，避免覆盖 exp 基线）")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_dir = os.path.join(args.data_dir, "train")
    target_dir = os.path.join(train_dir, "Target")
    baseline_dir = os.path.join(train_dir, "Baseline")
    li_dir = os.path.join(train_dir, "LI")
    mask_dir = os.path.join(train_dir, "Mask")
    phys_dir = os.path.join(train_dir, "Phys")
    meta_dir = os.path.join(args.data_dir, "meta")

    case_categories = None
    if args.case_categories:
        case_categories = [c.strip() for c in args.case_categories.split(",") if c.strip()]
    specs, metas = build_specs(meta_dir, split="train", holdout_patient=args.holdout_patient,
                               case_categories=case_categories)
    if case_categories:
        print(f"[data] case_category 过滤：只保留 {case_categories}，训练集 "
              f"{len(metas)} 样本 / {len(specs)} 序列", flush=True)
    if args.holdout_patient:
        print(f"[data] held-out 病人 {args.holdout_patient!r} 已排除，训练集 "
              f"{len(metas)} 样本 / {len(specs)} 序列（d_ref 也从训练集冻结）", flush=True)
    specs = [s for s in specs if len(s) >= args.min_z]
    if args.max_z > 0:
        specs = [s[: args.max_z] for s in specs]
    if args.max_implants > 0:
        specs = specs[: args.max_implants]
    print(f"[data] {len(specs)} 植入体序列 (min_z={args.min_z}, max_z={args.max_z}, "
          f"nb={args.nb}, device={args.device}, cuda={torch.cuda.is_available()})", flush=True)

    cache_name = "d_ref.npy" if not args.holdout_patient else f"d_ref_ho_{args.holdout_patient}.npy"
    d_ref = freeze_d_ref(metas, phys_dir, os.path.join(args.data_dir, cache_name))

    # 组 Z（序列长度）→ specs，保证同 batch 内 Z 一致（collate_theta 契约）。
    by_z = defaultdict(list)
    for spec in specs:
        by_z[len(spec)].append(spec)
    z_dist = {z: len(g) for z, g in sorted(by_z.items())}
    print(f"[data] 序列长度 Z 分布: {z_dist}", flush=True)

    loader = make_theta_sequence_loader(target_dir, baseline_dir, li_dir, mask_dir, phys_dir)

    # 每个 Z 一个 DataLoader（batch_size 截断到组内样本数）。
    dataloaders = {}
    for z, group in by_z.items():
        ds = ThetaDataset(group, loader=loader)
        bsz = min(args.batch_size, len(group))
        dataloaders[z] = DataLoader(ds, batch_size=bsz, shuffle=True,
                                    collate_fn=collate_theta, num_workers=args.num_workers)

    m_net = MNetwork(nb=args.nb, channels=args.channels,
                     use_velocity=not args.no_velocity,
                     gate_reads_dj=args.gate_reads_dj,
                     ablate_xm0=args.no_xm0, ablate_dj=args.no_dj, ablate_g=args.no_g).to(args.device)
    config = SSMConfig(nb=args.nb, channels=args.channels,
                       lambda_q=args.lambda_q, lambda_pc=args.lambda_pc,
                       lambda_down=args.lambda_down,
                       lambda_mu=args.lambda_mu,
                       trust_mode=args.trust_mode, recursion=args.recursion, d_ref=d_ref)
    trainer = SSMTrainer(m_net, config, device=args.device)   # 内部创建 FIND-Net
    # FIND-Net 与 M_μ 解耦学习率：FIND-Net 用官方 lr=2e-4，M_μ 用主 lr。
    opt = torch.optim.Adam([
        {"params": m_net.parameters(), "lr": args.lr},
        {"params": trainer.find_net.parameters(), "lr": args.findnet_lr},
    ], lr=args.lr)
    trainer.optimizer = opt
    n_params = sum(p.numel() for p in trainer.parameters())
    print(f"[model] MNetwork(nb={args.nb}, channels={args.channels}) + FINDNet "
          f"params={n_params} (M_μ lr={args.lr}, FIND-Net lr={args.findnet_lr})", flush=True)

    scheduler = None
    if args.lr_schedule == "cosine":
        warmup = max(args.warmup_epochs, 1)
        decay_steps = max(args.epochs - args.warmup_epochs, 1)

        def _lr_lambda(ep):
            if ep < args.warmup_epochs:
                return (ep + 1) / warmup
            t = (ep - args.warmup_epochs) / decay_steps
            return 0.5 * (1.0 + math.cos(math.pi * t))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
        print(f"[sched] cosine: warmup={args.warmup_epochs}ep, "
              f"decay over {args.epochs}ep -> eta_min~0", flush=True)

    suffix = f"_{args.trust_mode}"
    if args.holdout_patient:
        suffix += f"_ho_{args.holdout_patient}"
    if args.tag:
        suffix += f"_{args.tag}"

    for epoch in range(args.epochs):
        t0 = time.time()
        tot = q_tot = pc_tot = down_tot = mu_tot = 0.0
        n_batch = 0
        zs = list(dataloaders.keys())
        random.shuffle(zs)
        for z in zs:
            for batch in dataloaders[z]:
                batch = to_device(batch, args.device)
                total, q_loss, pc_loss, down_loss, mu_loss = trainer.train_step(
                    batch["thetas"], batch["X_clean"], batch["metal_mask"],
                    batch["x_hat"], clip_grad=args.clip_grad)
                if not torch.isfinite(total):
                    print(f"[NaN] Z={z} batch_idx={n_batch} total={total.item():.3f} "
                          f"Q={q_loss.item():.3f} pc={pc_loss.item():.3f} "
                          f"down={down_loss.item():.3f} "
                          f"mu={mu_loss.item():.3f}",
                          flush=True)
                    raise SystemExit(1)
                tot += float(total.item())
                q_tot += float(q_loss.item())
                pc_tot += float(pc_loss.item())
                down_tot += float(down_loss.item())
                mu_tot += float(mu_loss.item())
                n_batch += 1
        dt = time.time() - t0
        n = max(n_batch, 1)
        lr_now = opt.param_groups[0]["lr"]
        print(f"[epoch {epoch+1}/{args.epochs}] total={tot/n:.4f} Q={q_tot/n:.4f} "
              f"pc={pc_tot/n:.4f} down={down_tot/n:.4f} "
              f"mu={mu_tot/n:.4f} batches={n_batch} {dt:.1f}s lr={lr_now:.2e}", flush=True)
        if scheduler is not None:
            scheduler.step()
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            _out = os.path.join(args.data_dir,
                                f"ssm_nb{args.nb}{suffix}_e{epoch + 1}.pt")
            torch.save({"model": m_net.state_dict(),
                        "find_net": trainer.find_net.state_dict(),
                        "config": {"nb": args.nb, "channels": args.channels,
                                   "d_ref": d_ref, "epoch": epoch + 1,
                                   "use_velocity": not args.no_velocity,
                                   "gate_reads_dj": args.gate_reads_dj,
                                   "ablate_xm0": args.no_xm0, "ablate_dj": args.no_dj,
                                   "ablate_g": args.no_g}}, _out)
            print(f"[save] {_out}", flush=True)

    out = os.path.join(args.data_dir, f"ssm_nb{args.nb}{suffix}.pt")
    torch.save({"model": m_net.state_dict(),
                "find_net": trainer.find_net.state_dict(),
                "config": {"nb": args.nb, "channels": args.channels, "d_ref": d_ref,
                           "use_velocity": not args.no_velocity,
                           "gate_reads_dj": args.gate_reads_dj,
                           "ablate_xm0": args.no_xm0, "ablate_dj": args.no_dj,
                           "ablate_g": args.no_g}}, out)
    print(f"[save] {out}", flush=True)


if __name__ == "__main__":
    main()
