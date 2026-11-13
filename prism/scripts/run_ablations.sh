#!/usr/bin/env bash
# run_ablations.sh —— PRISM 消融矩阵执行脚本（基线 + A1-A4 + B1-B2 + C1 = 8 臂）。
#
# 铁律：所有消融在 Stage 2 进行，且考核「过渡区（Transition Zone）」指标为第一列。
# 前置：Stage 1 已产出 $STAGE1（findnet_stage1.pt，且用同一 --holdout-patient 训）。
#
# 每臂：Stage 2 交替训练（--out 独立 checkpoint）→ eval_mar.py 过渡区/全局指标。
# eval 自动从 checkpoint 读 recursion/gain_mode/findnet_lict 复现训练态，无需每臂覆盖。
#
# 用法（服务器 pzneuro 环境）：
#   DATA_DIR=/root/autodl-tmp/PRISM/training_data_astra \
#   STAGE1=$DATA_DIR/findnet_stage1.pt \
#   PATIENT=patients_head2 STEPS=2000 bash PRISM/scripts/run_ablations.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY=/root/miniconda3/envs/pzneuro/bin/python
DATA_DIR=${DATA_DIR:-/root/autodl-tmp/PRISM/training_data_astra}
STAGE1=${STAGE1:-$DATA_DIR/findnet_stage1.pt}
PATIENT=${PATIENT:-patients_head2}
STEPS=${STEPS:-2000}
OUTDIR=${OUTDIR:-$DATA_DIR/ablations}
mkdir -p "$OUTDIR"

# 公共参数（零泄漏：holdout=patients_head2 排除训练与 d_ref）
COMMON=(--data-dir "$DATA_DIR" --resume-findnet "$STAGE1" --steps "$STEPS"
        --batch-size-a 1 --batch-size-b 2 --cold-start-init obs
        --holdout-patient "$PATIENT" --lambda-mu 1.0 --num-workers 0)

run_arm() {
  local name=$1; shift
  local ckpt="$OUTDIR/$name.pt"
  local log="$OUTDIR/$name.log"
  echo "=== [$name] train ($(date)) ===" | tee "$log"
  "$PY" -u PRISM/scripts/stage2_alternating.py "${COMMON[@]}" --out "$ckpt" "$@" 2>&1 | tee -a "$log"
  echo "=== [$name] eval ($(date)) ===" | tee -a "$log"
  "$PY" -u PRISM/scripts/eval_mar.py --checkpoint "$ckpt" --data-dir "$DATA_DIR" \
      --patient "$PATIENT" --out-json "$OUTDIR/$name.eval.json" 2>&1 | tee -a "$log"
}

# 基线：全开（cold-start obs / 速度 ON / 卡尔曼增益 / 双向 / μ 先验 / L_pc ON）
run_arm 00_baseline

# A 组：递归与机制
run_arm A1_no_recursion  --recursion per_slice                       # 去递归（退化 2D）
run_arm A2_no_prior      --findnet-lict li                            # 去先验（μ 不喂 FIND-Net）
run_arm A3_gated_gain    --gain-mode gated                            # 增益→经验门控
run_arm A3_fixed_gain    --gain-mode fixed                            # 增益→常数
run_arm A4_no_velocity   --no-velocity                                # 去速度项

# B 组：物理约束
run_arm B1_hard_mask     --binarize-mask --loss-include-metal         # 硬切掩膜 + 金属内算 Loss
run_arm B2_no_phys       --lambda-pc 0                                # 去物理惩罚 L_pc

# C 组：极短序列
run_arm C1_force_short   --force-recursion-short                      # Z<3 强双向递归（预期发散）

echo "=== 全部完成。过渡区指标汇总见 $OUTDIR/*.eval.json ==="
