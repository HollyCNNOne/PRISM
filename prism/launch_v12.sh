#!/usr/bin/env bash
# launch_v12.sh —— v1.1-scale-fix 全量实验：Stage 1（fresh 预训练）+ Stage 2（9 臂消融矩阵）。
#
#   Stage 1：纯 2D FIND-Net 预训练（v1.1 配置：×255 + Mask 内部翻非金属），200 epoch 早停
#     ↓ 产出 findnet_stage1_v1.1.pt 后
#   Stage 2：run_ablations.sh（00_baseline + A1-A4 + B1-B2 + C1 = 9 臂），每臂 eval_mar.py
#     归档过渡区/分区指标 json + log_arm.py 写 v1.2-xxx 记录进 EXPERIMENT_LOG.md。
#
# 进 tmux 跑，断连无关。set -e：Stage 1 失败（NaN/崩溃）则整链终止，不跑 Stage 2。
set -euo pipefail

DATA_DIR=/root/autodl-tmp/PRISM/training_data_astra
PY=/root/miniconda3/envs/pzneuro/bin/python
HOLDOUT=patients_head2
ROOT=/root/autodl-tmp/prism_framework
cd "$ROOT"

LOG="$DATA_DIR/pipeline_v12.log"
STAGE1="$DATA_DIR/findnet_stage1_v1.1.pt"

# 0. 删除旧废权重（v1.0 塌缩网络的产物，绝不复用）
rm -f "$DATA_DIR/findnet_stage1.pt" "$DATA_DIR/findnet_stage1_e20.pt" \
      "$DATA_DIR"/findnet_stage1_epoch*.pt
echo "=== [v12] 已删除旧权重，开始 v1.1 全量 Stage 1（$(date)）===" | tee "$LOG"

# 1. Stage 1：fresh 预训练（无 --resume，从头训）
"$PY" -u PRISM/scripts/stage1_findnet.py \
    --data-dir "$DATA_DIR" --epochs 200 --batch-size 2 --lr 2e-4 \
    --holdout-patient "$HOLDOUT" --val-patient "$HOLDOUT" \
    --patience 10 --min-improvement 0.01 \
    --out "$STAGE1" \
    --device cuda 2>&1 | tee -a "$LOG"
echo "=== [v12] Stage 1 完成，最佳权重 $STAGE1（$(date)）===" | tee -a "$LOG"

# 2. Stage 2：9 臂消融矩阵（每臂 train → eval → log_arm 写 v1.2-xxx 记录）
echo "=== [v12] 启动 Stage 2 消融矩阵（9 臂）（$(date)）===" | tee -a "$LOG"
DATA_DIR="$DATA_DIR" STAGE1="$STAGE1" PATIENT="$HOLDOUT" \
  STEPS=2000 OUTDIR="$DATA_DIR/ablations_v12" \
  bash PRISM/scripts/run_ablations.sh 2>&1 | tee "$DATA_DIR/ablations_v12.log"
echo "=== [v12] 消融矩阵全部完成（$(date)）===" | tee -a "$LOG"
