#!/bin/bash
set -uo pipefail
DATA=/root/autodl-tmp/PRISM/training_data_astra
PY=/root/miniconda3/envs/pzneuro/bin/python
V22=/root/autodl-tmp/PRISM/v2.2
S1=$DATA/findnet_stage1_v2.1.pt
cd /root/autodl-tmp/prism_framework
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
check() { "$@"; rc=$?; if [ $rc -ne 0 ]; then echo "[FAIL] $*"; exit $rc; fi; }

echo "=== stage2 $(date) ==="
check "$PY" -u PRISM/scripts/stage2_alternating.py \
  --data-dir "$DATA" --steps 800 --batch-size-a 1 --batch-size-b 2 --max-z 8 \
  --eval-every 100 --stop-patience 3 --min-improvement 0.01 \
  --holdout-patient patients_head2 --resume-findnet "$S1" \
  --freq-record --freq-out "$V22/fre_image/fre_epoch/freq_stage2.csv" \
  --out "$DATA/ssm_stage2_v22.pt" --device cuda

echo "=== plot $(date) ==="
check "$PY" -u PRISM/scripts/plot_freq_epoch.py \
  --csv "$V22/fre_image/fre_epoch/freq_stage2.csv" \
  --out "$V22/fre_image/fre_epoch/freq_stage2.png" \
  --title "Stage 2 (FIND-Net + SSM) z-frequency over steps" --xlabel step

echo "=== 三路频域 $(date) ==="
check "$PY" -u PRISM/scripts/v22_freq_error.py \
  --data-dir "$DATA" --patient patients_head2 \
  --stage1 "$S1" --stage2 "$DATA/ssm_stage2_v22.pt" \
  --outdir "$V22/fre_image/fre_error" --device cuda

echo "=== STAGE2 ALL DONE $(date) ==="
touch "$V22/STAGE2_DONE"
