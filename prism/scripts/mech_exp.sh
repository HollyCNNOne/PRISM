#!/bin/bash
# mech_exp.sh — 机制实验（stage2 跑完后执行）：exp007 内容自适应极点 / exp008 频率加权
# 公平性：两个实验臂都用 **baseline stage2 实际跑过的步数**，且关闭早停（同一预算）。
set -uo pipefail
DATA=/root/autodl-tmp/PRISM/training_data_astra
PY=/root/miniconda3/envs/pzneuro/bin/python
V22=/root/autodl-tmp/PRISM/v2.2
S1=$DATA/findnet_stage1_v2.1.pt
BASE=$DATA/ssm_stage2_v22.pt
cd /root/autodl-tmp/prism_framework
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
check() { "$@"; rc=$?; if [ $rc -ne 0 ]; then echo "[FAIL] $*"; exit $rc; fi; }

echo "=== 等待 baseline stage2 完成 $(date) ==="
while [ ! -f "$V22/STAGE2_DONE" ]; do sleep 120; done

STEPS=$("$PY" - <<'PYEOF'
import torch
ck = torch.load("/root/autodl-tmp/PRISM/training_data_astra/ssm_stage2_v22.pt",
                map_location="cpu", weights_only=False)
print(int(ck.get("config", {}).get("step", 800)))
PYEOF
)
echo "=== baseline 步数 = $STEPS（两臂同预算） $(date) ==="

run_arm () {   # $1=名字  $2..=额外参数
  local name=$1; shift
  echo "=== $name 开始 $(date) ==="
  check "$PY" -u PRISM/scripts/stage2_alternating.py \
    --data-dir "$DATA" --steps "$STEPS" --batch-size-a 1 --batch-size-b 2 --max-z 8 \
    --eval-every 100 --stop-patience 3 --min-improvement 0.01 \
    --holdout-patient patients_head2 --resume-findnet "$S1" \
    --freq-record --freq-out "$V22/fre_image/fre_epoch/freq_$name.csv" \
    --out "$DATA/ssm_$name.pt" --device cuda "$@"
  check "$PY" -u PRISM/scripts/plot_freq_epoch.py \
    --csv "$V22/fre_image/fre_epoch/freq_$name.csv" \
    --out "$V22/fre_image/fre_epoch/freq_$name.png" \
    --title "Stage 2 — $name" --xlabel step
  check "$PY" -u PRISM/scripts/v22_freq_error.py \
    --data-dir "$DATA" --patient patients_head2 \
    --stage1 "$S1" --stage2 "$DATA/ssm_$name.pt" \
    --outdir "$V22/fre_image/fre_error_$name" --device cuda
  check "$PY" -u PRISM/scripts/v22_metrics_table.py --stage2 "$DATA/ssm_$name.pt" \
    --outdir "$V22/info_loss" --tag "$name" 2>/dev/null || true
  echo "=== $name 完成 $(date) ==="
}

run_arm exp007_zcurv --use-z-curvature
run_arm exp008_zfreq --zfreq-gain 1.0

echo "=== 机制实验全部完成 $(date) ==="
touch "$V22/MECH_DONE"
