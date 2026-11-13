#!/usr/bin/env bash
# watch_v12.sh —— v1.1 全量管线监控（服务端 tmux 内跑，断连无关）。
# 每 600s 轮询 pipeline_v12.log 的阶段，追加一行到 watch_v12.log；终态（DONE/FAIL）后退出。
LOG=/root/autodl-tmp/PRISM/training_data_astra/pipeline_v12.log
WATCH=/root/autodl-tmp/prism_framework/watch_v12.log

echo "[watch] start $(date +%m-%d_%H:%M)" >> "$WATCH"
for i in $(seq 1 1440); do   # 最多 1440×600s = 10 天
  if grep -q "消融矩阵全部完成" "$LOG" 2>/dev/null; then S=DONE;
  elif grep -qE "Traceback|\[NaN\]|SystemExit" "$LOG" 2>/dev/null; then S=FAIL;
  elif ps -eo args | grep -q "[s]tage2_alternating"; then S=ABLATION;
  elif ps -eo args | grep -q "[s]tage1_findnet"; then S=STAGE1;
  else S=UNKNOWN; fi
  last=$(grep -E "epoch [0-9]+/|step [0-9]+/|Stage 1 完成|early-stop|全部完成|\[v12\]" "$LOG" 2>/dev/null | tail -1)
  echo "[watch $(date +%m-%d_%H:%M)] $S | $last" >> "$WATCH"
  case "$S" in
    DONE|FAIL) echo "[watch] 终态 $S，退出" >> "$WATCH"; exit 0;;
  esac
  sleep 600
done
echo "[watch] TIMEOUT 10 天仍在跑" >> "$WATCH"
