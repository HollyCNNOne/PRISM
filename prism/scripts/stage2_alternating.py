#!/usr/bin/env python
"""stage2_alternating.py —— Stage 2：交替训练（联合 SSM+FIND-Net vs FIND-Net-only）。

数据分层（按 case_category × 序列长度，缺一不可）：
  A（联合 3D，SSM+FIND-Net）：complete + large_residue + medium_residue 中 Z>=MIN_REC_Z 的
    序列（序列够长，值得启动递归）。训练步走 trainer.train_step 全损失（L_down+L_mu+L_Q+L_pc），
    更新 SSM+FIND-Net。
  B（2D，仅 FIND-Net）：small_residue 全部单切片 + A 类别里 Z<MIN_REC_Z 的过短序列（降级 2D，
    不强行启动 SSM 递归）。训练步只 FIND-Net masked-MSE，更新 FIND-Net。

每走一步交替：偶数步 A、奇数步 B。单一优化器（含两组参数），B 步只产 FIND-Net 梯度，
m_net 零梯度不更新，从而保护 SSM 连续性梯度不被碎片化单切片污染。

冷启动暖启动（--cold-start-init obs）：S0.μ 用首片 Baseline 观测而非全零，强拉短序列收敛。
  "obs"=Baseline 脏图（组织观测，推荐）；"xm0"=金属先验 x_M0（非组织量，慎用）；"zero"=现状。

用法（服务器 pzneuro 环境）：
    /root/miniconda3/envs/pzneuro/bin/python -u scripts/stage2_alternating.py \
        --data-dir /root/autodl-tmp/PRISM/training_data_astra \
        --resume-findnet /root/autodl-tmp/PRISM/training_data_astra/findnet_stage1.pt \
        --steps 2000 --batch-size-a 1 --batch-size-b 2 --cold-start-init obs
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

_FRAMEWORK_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _FRAMEWORK_ROOT)

from prism.data.dataset import ThetaDataset, collate_theta, make_theta_sequence_loader
from prism.data.dataset2d import Slice2DDataset, collate2d
from prism.models.m_network import MNetwork
from prism.models.find_net import FINDNet, FINDNetArgs
from prism.models.find_net_gf import FINDNet as FINDNetGF, FINDNetArgs as FINDNetArgsGF
from prism.models.r_theta import HU_SCALE
from prism.training.trainer import SSMTrainer
from prism.utils.config import SSMConfig


def _sample_idx(sid):
    return int(sid.rsplit("_", 1)[1])


def load_metas(meta_dir, holdout_patient=None):
    """读训练 split 的 meta；holdout_patient 非 None 时排除该病人（零泄漏，d_ref 也只用训练集）。"""
    metas = {}
    for f in glob.glob(os.path.join(meta_dir, "*.json")):
        m = json.load(open(f))
        if m.get("split", "train") != "train":
            continue
        if holdout_patient and m.get("patient_id") == holdout_patient:
            continue
        metas[m["sample_id"]] = m
    return metas


def split_specs(metas, a_categories, min_rec_z, force_recursion_short=False):
    """按 sequence_id 分组，再按 case_category × 长度分 A（3D）与 B（2D 单切片）。

    A = 类别 ∈ a_categories 且 len(seq) >= min_rec_z 的序列；其余全部 → B 的单切片。
    force_recursion_short（C1）：a_categories 里 Z<min_rec_z 的过短序列不再降级 2D，
    强行进 A 双向递归（预期 Z=1,2 在过渡区发散，证降级策略必须）。
    """
    by_seq = defaultdict(list)
    for sid, m in metas.items():
        by_seq[m["sequence_id"]].append(sid)
    a_specs, b_ids = [], []
    for seq in by_seq.values():
        seq = sorted(seq, key=_sample_idx)
        cat = metas[seq[0]]["case_category"]
        long_enough = len(seq) >= min_rec_z
        if cat in a_categories and (long_enough or force_recursion_short):
            a_specs.append(seq)
        else:
            b_ids.extend(seq)
    a_specs.sort(key=lambda s: (_sample_idx(s[0]), len(s)))
    b_ids.sort(key=_sample_idx)
    return a_specs, b_ids


def freeze_d_ref(sample_ids, phys_dir, cache_path, stride=4):
    """d_ref = median_j d_j（训练集一次冻结，同 run_train.freeze_d_ref）。"""
    if os.path.exists(cache_path):
        return float(np.load(cache_path)[0])
    chunks = [np.load(os.path.join(phys_dir, f"{sid}_d_j.npy"))[0, ::stride, ::stride].ravel()
              for sid in sample_ids]
    d_ref = float(np.median(np.concatenate(chunks).astype(np.float64)))
    np.save(cache_path, np.array([d_ref], dtype=np.float64))
    return d_ref


def to_device(batch, device):
    out = {
        "thetas": [{k: v.to(device) for k, v in t.items()} for t in batch["thetas"]],
        "X_clean": batch["X_clean"].to(device),
        "metal_mask": batch["metal_mask"].to(device),
        "x_hat": batch["x_hat"].to(device),
        "meta": batch["meta"],
    }
    if "li" in batch:
        out["li"] = batch["li"].to(device)
    return out


def masked_mse(x_hat, target, mask):
    resid = (x_hat - target) / HU_SCALE
    se = resid ** 2
    m = (1.0 - mask)
    return (se * m).sum() / (m.sum() + 1e-12)


def cycle(dl):
    while True:
        for b in dl:
            yield b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/root/autodl-tmp/PRISM/training_data_astra")
    ap.add_argument("--nb", type=int, default=1)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size-a", type=int, default=1)
    ap.add_argument("--batch-size-b", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--findnet-lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-rec-z", type=int, default=3)
    ap.add_argument("--a-categories", default="complete,large_residue,medium_residue")
    ap.add_argument("--b-categories", default="small_residue")
    ap.add_argument("--cold-start-init", default="obs", choices=["zero", "obs", "xm0"])
    ap.add_argument("--holdout-patient", default=None,
                    help="零泄漏切分：训练/冻结 d_ref 时排除该病人（如 patients_head2，最终盲测）")
    ap.add_argument("--lambda-q", type=float, default=1.0)
    ap.add_argument("--lambda-pc", type=float, default=0.1)
    ap.add_argument("--lambda-down", type=float, default=1.0)
    ap.add_argument("--lambda-mu", type=float, default=1.0)
    ap.add_argument("--lambda-ssim", type=float, default=0.1,
                    help="v1.3：L_down 内 SSIM 高频结构约束权重（0=关）")
    ap.add_argument("--backfill-r", type=int, default=8,
                    help="v1.3：最终输出金属回填膨胀半径 px（0=关）")
    ap.add_argument("--recursion", default="bidirectional",
                    choices=["bidirectional", "forward", "per_slice"])
    # ---- 消融矩阵开关（默认 = 基线全开） ----
    ap.add_argument("--no-velocity", action="store_true",
                    help="A4 去速度：速度项 Δμ 不进 backbone/M_μ（置 0），纯位置模型")
    ap.add_argument("--gain-mode", default="kalman", choices=["kalman", "gated", "fixed"],
                    help="A3 增益替换：kalman=物理卡尔曼增益（现状）/ gated=经验门控 g(d_j) / "
                         "fixed=可学习常数 γ")
    ap.add_argument("--findnet-lict", default="mu", choices=["mu", "li"],
                    help="A2 去先验：FIND-Net LIct 输入 mu=SSM 先验（现状）/ li=经典 LI-MAR")
    ap.add_argument("--binarize-mask", action="store_true",
                    help="B1 硬切掩膜：金属掩膜二值化 0/1（默认连续 vf）")
    ap.add_argument("--mask-bin-thr", type=float, default=0.5,
                    help="B1 二值化阈值（vf > thr → 金属）")
    ap.add_argument("--loss-include-metal", action="store_true",
                    help="B1 金属内算 Loss：金属区也进重建损失（默认 (1-mask) 弃疗金属）")
    ap.add_argument("--force-recursion-short", action="store_true",
                    help="C1 强递归：Z<min_rec_z 的过短序列不降级 2D，强行进 A 双向递归")
    ap.add_argument("--clip-grad", type=float, default=0.0)
    ap.add_argument("--max-z", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--out", default=None,
                    help="checkpoint 输出路径（None=默认 data_dir/ssm_stage2.pt；消融矩阵每臂一个）")
    ap.add_argument("--resume-findnet", default=None,
                    help="Stage 1 产出的 find_net checkpoint（暖启动）")
    ap.add_argument("--gf", action="store_true",
                    help="v2.2：FIND-Net 用 GF 版（FourierUnit 加可学习 Gaussian 带通）")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_dir = os.path.join(args.data_dir, "train")
    metas = load_metas(os.path.join(args.data_dir, "meta"), args.holdout_patient)
    a_cats = {c.strip() for c in args.a_categories.split(",") if c.strip()}
    b_cats = {c.strip() for c in args.b_categories.split(",") if c.strip()}
    a_specs, b_ids = split_specs(metas, a_cats, args.min_rec_z,
                                 args.force_recursion_short)
    if args.max_z > 0:
        a_specs = [s[:args.max_z] for s in a_specs]
    print(f"[data] holdout={args.holdout_patient!r}  "
          f"A(3D joint)={len(a_specs)} seq (cats={sorted(a_cats)}, Z>={args.min_rec_z}), "
          f"B(2D FIND-Net-only)={len(b_ids)} slices (cats={sorted(b_cats)} + short-Z)",
          flush=True)

    d_ref_cache = ("d_ref.npy" if not args.holdout_patient
                   else f"d_ref_ho_{args.holdout_patient}.npy")
    d_ref = freeze_d_ref(list(metas.keys()), os.path.join(train_dir, "Phys"),
                         os.path.join(args.data_dir, d_ref_cache))
    print(f"[d_ref] {d_ref:.6g} (训练集 {len(metas)} 样本，排除 {args.holdout_patient!r})",
          flush=True)

    # A：3D 序列 DataLoader（按 Z 分组，collate_theta 要求同 Z）
    seq_loader = make_theta_sequence_loader(
        os.path.join(train_dir, "Target"),
        os.path.join(train_dir, "Baseline"),
        os.path.join(train_dir, "LI"),
        os.path.join(train_dir, "Mask"),
        os.path.join(train_dir, "Phys"))
    by_z = defaultdict(list)
    for s in a_specs:
        by_z[len(s)].append(s)
    dl_a = {}
    for z, group in by_z.items():
        ds = ThetaDataset(group, loader=seq_loader)
        dl_a[z] = DataLoader(ds, batch_size=min(args.batch_size_a, len(group)),
                             shuffle=True, collate_fn=collate_theta,
                             num_workers=args.num_workers)

    # B：2D 单切片 DataLoader
    ds_b = Slice2DDataset(
        b_ids,
        os.path.join(train_dir, "Target"),
        os.path.join(train_dir, "Baseline"),
        os.path.join(train_dir, "LI"),
        os.path.join(train_dir, "Mask"),
        augment=True)
    dl_b = DataLoader(ds_b, batch_size=args.batch_size_b, shuffle=True,
                      collate_fn=collate2d, num_workers=args.num_workers, drop_last=True)

    m_net = MNetwork(nb=args.nb, channels=args.channels,
                     use_velocity=not args.no_velocity).to(args.device)
    find_net = (FINDNetGF(FINDNetArgsGF()) if args.gf else FINDNet(FINDNetArgs())).to(args.device)
    if args.resume_findnet:
        ck = torch.load(args.resume_findnet, map_location=args.device, weights_only=False)
        find_net.load_state_dict(ck["find_net"])
        print(f"[load] find_net from {args.resume_findnet} (epoch {ck['config'].get('epoch')})",
              flush=True)

    config = SSMConfig(nb=args.nb, channels=args.channels,
                       lambda_q=args.lambda_q, lambda_pc=args.lambda_pc,
                       lambda_down=args.lambda_down, lambda_mu=args.lambda_mu,
                       lambda_ssim=args.lambda_ssim,
                       backfill_r=args.backfill_r,
                       recursion=args.recursion, d_ref=d_ref,
                       cold_start_init=args.cold_start_init,
                       gain_mode=args.gain_mode,
                       findnet_lict_source=args.findnet_lict,
                       binarize_mask=args.binarize_mask,
                       mask_bin_thr=args.mask_bin_thr,
                       loss_include_metal=args.loss_include_metal)
    trainer = SSMTrainer(m_net, config, device=args.device, find_net=find_net)
    opt = torch.optim.Adam([
        {"params": m_net.parameters(), "lr": args.lr},
        {"params": find_net.parameters(), "lr": args.findnet_lr},
    ], lr=args.lr)
    trainer.optimizer = opt
    print(f"[model] MNetwork(nb={args.nb}) + FINDNet, cold_start_init={args.cold_start_init}, "
          f"recursion={args.recursion} use_velocity={not args.no_velocity} "
          f"gain_mode={args.gain_mode} findnet_lict={args.findnet_lict} "
          f"binarize={args.binarize_mask} loss_metal={args.loss_include_metal} "
          f"force_short={args.force_recursion_short}", flush=True)

    def cycle_a():
        # 随机轮转 Z 组 + 组内循环，保证交替时跨 Z 打散
        while True:
            zs = list(dl_a.keys())
            random.shuffle(zs)
            for z in zs:
                for b in dl_a[z]:
                    yield b

    it_a = cycle_a()
    it_b = cycle(dl_b)

    out = args.out or os.path.join(args.data_dir, "ssm_stage2.pt")

    def save_cfg(step):
        """checkpoint config：完整记录消融开关，eval_mar 据此复现训练态。"""
        return {"nb": args.nb, "channels": args.channels, "d_ref": d_ref, "step": step,
                "cold_start_init": args.cold_start_init,
                "holdout_patient": args.holdout_patient,
                "use_velocity": not args.no_velocity, "recursion": args.recursion,
                "gain_mode": args.gain_mode, "findnet_lict_source": args.findnet_lict,
                "binarize_mask": args.binarize_mask, "mask_bin_thr": args.mask_bin_thr,
                "loss_include_metal": args.loss_include_metal,
                "lambda_ssim": args.lambda_ssim,
                "backfill_r": args.backfill_r}

    a_tot = {"total": 0.0, "q": 0.0, "pc": 0.0, "down": 0.0, "mu": 0.0}
    b_tot = 0.0
    na = nb = 0
    for step in range(args.steps):
        if step % 2 == 0:
            # A：联合 SSM+FIND-Net（train_step 全损失，更新两组参数）
            batch = to_device(next(it_a), args.device)
            total, q, pc, down, mu = trainer.train_step(
                batch["thetas"], batch["X_clean"], batch["metal_mask"],
                batch["x_hat"], clip_grad=args.clip_grad, li=batch.get("li"))
            if not torch.isfinite(total):
                print(f"[NaN] step={step} A total={total.item()}", flush=True)
                raise SystemExit(1)
            a_tot["total"] += float(total.item())
            a_tot["q"] += float(q.item())
            a_tot["pc"] += float(pc.item())
            a_tot["down"] += float(down.item())
            a_tot["mu"] += float(mu.item())
            na += 1
        else:
            # B：仅 FIND-Net（单切片 masked-MSE，m_net 零梯度不更新）
            ct_ma, lict, mask, target = next(it_b)
            ct_ma = ct_ma.to(args.device)
            lict = lict.to(args.device)
            mask = mask.to(args.device)
            target = target.to(args.device)
            opt.zero_grad()
            x_hat = find_net.reconstruct(ct_ma, lict, mask)
            loss = masked_mse(x_hat, target, mask)
            if not torch.isfinite(loss):
                print(f"[NaN] step={step} B loss={loss.item()}", flush=True)
                raise SystemExit(1)
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(find_net.parameters(), args.clip_grad)
            opt.step()
            b_tot += float(loss.item())
            nb += 1

        if (step + 1) % 200 == 0:
            print(f"[step {step + 1}/{args.steps}] "
                  f"A total={a_tot['total'] / max(na, 1):.4f} "
                  f"down={a_tot['down'] / max(na, 1):.4f} "
                  f"mu={a_tot['mu'] / max(na, 1):.4f} (na={na}) | "
                  f"B down={b_tot / max(nb, 1):.4f} (nb={nb})", flush=True)
            a_tot = {"total": 0.0, "q": 0.0, "pc": 0.0, "down": 0.0, "mu": 0.0}
            b_tot = 0.0
            na = nb = 0

        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            torch.save({"model": m_net.state_dict(), "find_net": find_net.state_dict(),
                        "config": save_cfg(step + 1)}, out)
            print(f"[save] {out}", flush=True)

    torch.save({"model": m_net.state_dict(), "find_net": find_net.state_dict(),
                "config": save_cfg(args.steps)}, out)
    print(f"[save] {out}", flush=True)


if __name__ == "__main__":
    main()
