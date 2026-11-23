#!/bin/bash
# mech_exp2.sh — 第二个机制实验：exp008 频率加权训练（Sobolov-z），单独一臂
# 与 exp007 同一协议：同预算（= baseline 实际步数）、同早停、同 held-out 监控。
# 末尾自动跑多轴对比 baseline vs exp008_zfreq。
set -uo pipefail
DATA=/root/autodl-tmp/PRISM/training_data_astra
PY=/root/miniconda3/envs/pzneuro/bin/python
V22=/root/autodl-tmp/PRISM/v2.2
S1=$DATA/findnet_stage1_v2.1.pt
BASE=$DATA/ssm_stage2_v22.pt
cd /root/autodl-tmp/prism_framework
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
check() { "$@"; rc=$?; if [ $rc -ne 0 ]; then echo "[FAIL] $*"; exit $rc; fi; }

STEPS=$("$PY" - <<'PYEOF'
import torch
ck = torch.load("/root/autodl-tmp/PRISM/training_data_astra/ssm_stage2_v22.pt",
                map_location="cpu", weights_only=False)
print(int(ck.get("config", {}).get("step", 800)))
PYEOF
)
echo "=== exp008 开始 $(date)  预算步数=$STEPS ==="

check "$PY" -u PRISM/scripts/stage2_alternating.py \
  --data-dir "$DATA" --steps "$STEPS" --batch-size-a 1 --batch-size-b 2 --max-z 8 \
  --eval-every 100 --stop-patience 3 --min-improvement 0.01 \
  --holdout-patient patients_head2 --resume-findnet "$S1" \
  --freq-record --freq-out "$V22/fre_image/fre_epoch/freq_exp008_zfreq.csv" \
  --out "$DATA/ssm_exp008_zfreq.pt" --device cuda --zfreq-gain 1.0
echo "=== exp008 训练完成 $(date) ==="

check "$PY" -u PRISM/scripts/plot_freq_epoch.py \
  --csv "$V22/fre_image/fre_epoch/freq_exp008_zfreq.csv" \
  --out "$V22/fre_image/fre_epoch/freq_exp008_zfreq.png" \
  --title "Stage 2 — exp008_zfreq" --xlabel step

echo "=== 多轴对比 baseline vs exp008 $(date) ==="
check "$PY" -u prism/scripts/compare_arms.py \
  --ckpts "$BASE,$DATA/ssm_exp008_zfreq.pt" --names baseline,exp008_zfreq \
  --max-seq 6 --outdir "$V22/arm_compare_exp008" --device cuda

echo "=== EXP008 DONE $(date) ==="
touch "$V22/EXP008_DONE"
