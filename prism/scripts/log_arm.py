#!/usr/bin/env python
"""log_arm.py —— 每臂跑完后按《实验与版本记录规范》10 项模板写入 v1.3-xxx 记录。

倒序置顶：新记录插到 EXPERIMENT_LOG.md 标题头之后（首个 "---" 分隔线之后）。
客观指标（过渡区/三分区/全局 RMSE/PSNR/peak/植入体数）自动从 eval json 读入；
分析字段（成功点/失败点/根因）留占位，待 9 臂矩阵跑完后统一回填。

用法（run_ablations.sh 每臂 eval 后调用）：
    python PRISM/scripts/log_arm.py --name 00_baseline --json <eval.json> \
        --log /root/autodl-tmp/prism_framework/EXPERIMENT_LOG.md
"""
import argparse
import datetime
import json
import os

ARM_META = {
    "00_baseline": {
        "purpose": "基线全开：卡尔曼增益 + 双向递归 + SSM 先验 + 速度项 + L_pc。",
        "method": "cold-start=obs / recursion=bidirectional / gain=kalman / findnet_lict=mu / velocity=on / L_pc=on",
        "root_cause": "（待回填）",
    },
    "A1_no_recursion": {
        "purpose": "去递归：per_slice 退化 2D，验证递归先验的贡献。",
        "method": "--recursion per_slice",
        "root_cause": "（待回填）",
    },
    "A2_no_prior": {
        "purpose": "去先验：FIND-Net 不吃 SSM μ，改用经典 LI-MAR。",
        "method": "--findnet-lict li",
        "root_cause": "（待回填）",
    },
    "A3_gated_gain": {
        "purpose": "增益替换：卡尔曼 → 经验门控 g(d_j)。",
        "method": "--gain-mode gated",
        "root_cause": "（待回填）",
    },
    "A3_fixed_gain": {
        "purpose": "增益替换：卡尔曼 → 可学习常数 γ。",
        "method": "--gain-mode fixed",
        "root_cause": "（待回填）",
    },
    "A4_no_velocity": {
        "purpose": "去速度：Δμ 不进 backbone/M_μ（纯位置模型）。",
        "method": "--no-velocity",
        "root_cause": "（待回填）",
    },
    "B1_hard_mask": {
        "purpose": "硬切掩膜 + 金属内算 Loss（验证连续 vf 掩膜的贡献）。",
        "method": "--binarize-mask --loss-include-metal",
        "root_cause": "（待回填）",
    },
    "B2_no_phys": {
        "purpose": "去物理惩罚 L_pc。",
        "method": "--lambda-pc 0",
        "root_cause": "（待回填）",
    },
    "C1_force_short": {
        "purpose": "极短序列（Z<3）强双向递归（预期发散，证降级策略必须）。",
        "method": "--force-recursion-short",
        "root_cause": "（待回填）",
    },
}


def _fmt(region):
    r = region or {}
    rmse = r.get("rmse", float("nan"))
    bias = r.get("bias", float("nan"))
    psnr = r.get("psnr", float("nan"))
    return f"{rmse:.2f}", f"{bias:.1f}", f"{psnr:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--log", default="/root/autodl-tmp/prism_framework/EXPERIMENT_LOG.md")
    args = ap.parse_args()

    meta = ARM_META.get(args.name, {"purpose": "（待回填）", "method": "", "root_cause": "（待回填）"})
    d = json.load(open(args.json))
    res = d.get("results", {})
    tz = _fmt(res.get("过渡区"))
    full = _fmt(res.get("充分"))
    part = _fmt(res.get("部分阴影"))
    shadow = _fmt(res.get("完全阴影"))
    glob = _fmt(res.get("全局"))
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    record = (
        f"## v1.3-{args.name}\n\n"
        f"- [版本号] v1.3-{args.name}\n"
        f"- [时间] {ts}\n"
        f"- [实验前提与条件] v1.3-baseline 定案（MSE+0.1·SSIM + 回填 r8）+ findnet_stage1_v1.1.pt + "
        f"Stage 2 消融矩阵（holdout=patients_head2 零泄漏，d_ref=4.10023e+06 冻结）。\n"
        f"- [本次目的] {meta['purpose']}\n"
        f"- [采用方法] {meta['method']}\n"
        f"- [实验结果] 过渡区 RMSE={tz[0]}/Bias={tz[1]}/PSNR={tz[2]}；充分区 RMSE={full[0]}；"
        f"部分阴影 RMSE={part[0]}；完全阴影 RMSE={shadow[0]}；全局 RMSE={glob[0]}"
        f"（peak={d.get('peak_hu', float('nan')):.0f} HU，{d.get('n_implants', '?')} 植入体）。\n"
        f"- [成功点] （待回填：与 00_baseline 过渡区对比）\n"
        f"- [失败点/异常] （待回填：确认无 NaN / 无 OOM 崩溃）\n"
        f"- [问题根因分析] {meta['root_cause']}\n"
        f"- [下一步状态] 待 9 臂矩阵跑完后统一定案。\n"
    )

    if os.path.exists(args.log):
        with open(args.log, encoding="utf-8") as f:
            content = f.read()
    else:
        content = "# PRISM 实验与版本记录（EXPERIMENT_LOG）\n\n> 最新在前（倒序）。\n\n---\n\n"

    # 倒序置顶：插到标题头之后（首个 "---" 分隔线之后）
    marker = "---"
    pos = content.find(marker)
    if pos == -1:
        content = record + "\n---\n\n" + content
    else:
        nl = content.find("\n", pos)
        insert_at = nl + 1 if nl != -1 else len(content)
        content = content[:insert_at] + "\n" + record + "\n---\n" + content[insert_at:]

    with open(args.log, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[log_arm] v1.3-{args.name} 已写入 {args.log}", flush=True)


if __name__ == "__main__":
    main()
