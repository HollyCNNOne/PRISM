#!/bin/bash
# v22_pipeline.sh — V2.2 唯一合法链（数据契约 → 训练 → 数值实验）每一步失败即停
set -uo pipefail
DATA=/root/autodl-tmp/PRISM/training_data_astra
PY=/root/miniconda3/envs/pzneuro/bin/python
AGIR=/root/miniconda3/envs/agir/bin/python
ROOT=/root/autodl-tmp/prism_framework
COIL=/root/autodl-tmp/PRISM/3D_Coil
HOLDOUT=patients_head2
V22=/root/autodl-tmp/PRISM/v2.2
S1=$DATA/findnet_stage1_v2.1.pt

cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
check() { "$@"; rc=$?; if [ $rc -ne 0 ]; then echo "[FAIL] $*"; exit $rc; fi; }

# ---------- P0.3 数据契约 ----------
if [ ! -f "$DATA/REGEN_PHYS_DONE" ]; then
  echo "=== P0.3 regen_skimage + regen_phys $(date) ==="
  cd "$COIL"
  "$AGIR" -u regen_skimage.py --nproc 16 > "$DATA/regen_v22_skimage.log" 2>&1
  "$AGIR" -u regen_phys.py   --nproc 16 > "$DATA/regen_v22_phys.log" 2>&1
  touch "$DATA/REGEN_PHYS_DONE"
  cd "$ROOT"
fi

# ---------- P0.4 三关验证 ----------
echo "=== P0.4 三关验证 $(date) ==="
check "$PY" -u PRISM/scripts/verify_dataset.py --data-dir "$DATA" --findnet "$S1" --outdir "$V22/verify"

# ---------- P2.1 信息缺失度 ----------
echo "=== P2.1 观测端信息缺失度 $(date) ==="
check "$AGIR" -u PRISM/scripts/v22_obs_info_loss.py --data-dir "$DATA" --outdir "$V22/info_loss"

# ---------- P2.2 FIND-Net 四切片 ----------
echo "=== P2.2 FIND-Net 四切片 $(date) ==="
check "$PY" -u PRISM/scripts/v22_findnet_obs_error.py --data-dir "$DATA" --ckpt "$S1" --outdir "$V22/info_loss"

# ---------- 四尺寸对比图（含回填） ----------
echo "=== P2.3 四尺寸对比图 $(date) ==="
check "$PY" -u PRISM/scripts/viz_four_sizes_correct.py

# ---------- P1.2 stage2 ----------
echo "=== P1.2 stage2 (No-GF, max-z 8, 收敛即停) $(date) ==="
check "$PY" -u PRISM/scripts/stage2_alternating.py \
  --data-dir "$DATA" --steps 1000 --batch-size-a 1 --batch-size-b 2 --max-z 8 \
  --eval-every 100 --stop-patience 3 --min-improvement 0.01 \
  --holdout-patient "$HOLDOUT" --resume-findnet "$S1" \
  --freq-record --freq-out "$V22/fre_image/fre_epoch/freq_stage2.csv" \
  --out "$DATA/ssm_stage2_v22.pt" --device cuda

check "$PY" -u PRISM/scripts/plot_freq_epoch.py \
  --csv "$V22/fre_image/fre_epoch/freq_stage2.csv" \
  --out "$V22/fre_image/fre_epoch/freq_stage2.png" \
  --title "Stage 2 (FIND-Net + SSM) z-frequency over steps" --xlabel step

# ---------- P2.4 三路频域 ----------
echo "=== P2.4 三路频域 $(date) ==="
check "$PY" -u PRISM/scripts/v22_freq_error.py \
  --data-dir "$DATA" --patient "$HOLDOUT" \
  --stage1 "$S1" --stage2 "$DATA/ssm_stage2_v22.pt" \
  --outdir "$V22/fre_image/fre_error" --device cuda

echo "=== ALL DONE $(date) ===" | tee -a "$V22/pipeline_v22.log"
touch "$V22/PIPELINE_DONE"
