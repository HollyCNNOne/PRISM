# V2.2-baseline-SSM-2

> **本文档的职责（三份文档分工）**
> - `docs/RESEARCH_LOG.md` = **科研日志**：只写分析、推理、判断、对论文的含义
> - `docs/experiments/<编号>/README.md` = **实验日志（本文）**：只写怎么做的、产物在哪、怎么复现、状态与问题
> - `docs/DATA_LOG.md` = **数据日志**：只写数值、索引、口径、验证、作废记录

| 项 | 值 |
|---|---|
| 编号 | V2.2-baseline-SSM-2（原 `exp008_freq_weighted_training`） |
| 状态 | ✅ 定案 |
| 数据层 | 解析仿真（平行束 720 视角 / PSF σ=6.0） |
| 一句话 | 频率加权臂：**21 项里赢 14 项，唯独没赢靶心**——near 的 σ=1 相关 0.105→0.104 |

---

## 1. 对象与问题

> 在 L_down 里按 GT 的 z 高频加权，能否让输出保留更多高频结构？

对照臂：`baseline`（标准 L_down）。**两臂只差 `zfreq_gain`（0 → 1.0）**，其余全部相同。

## 2. 环境

| 项 | 值 |
|---|---|
| 运行位置 | AutoDL 服务器 `connect.westb.seetacloud.com:13686` |
| 设备 | GPU RTX 3080 Ti（12 GB） |
| conda env | `pzneuro`（训练 / 推理） |

## 3. 配置

| 参数 | baseline | exp008 |
|---|---|---|
| `zfreq_gain` | 0.0 | **1.0** |
| 网络/递归 | nb=1 / channels=32 / kalman / bidirectional / max-z 8 | 完全相同 |
| batch | A=1, B=2 | 完全相同 |
| λ_q / λ_pc / λ_down / λ_mu | 1.0 / 0.1 / 1.0 / 1.0 | 同 |
| λ_ssim / λ_grad | 0.1 / 0.1 | 同 |
| 预算 | 800 步 | 800 步 |
| 早停 | patience 3 / 最小改进 1% | 同 |

**加权公式**（`losses.downstream_reconstruction_parts`）：

```
w = 1 + zfreq_gain · clamp(|Δz GT| / 50, max=4)
mask = mask · w
```

即 GT 沿 z 变化大的体素误差权重更高，**上限 5 倍**。

快照：`config/args.txt`

## 4. 数据

| 项 | 值 |
|---|---|
| 来源 | `/root/autodl-tmp/PRISM/training_data_astra/` |
| 训练 | A（3D 联合）= 12 seq；B（2D FIND-Net-only）= 21 slices |
| held-out 监控 | `patients_head2`，6 条长序列 |
| d_ref | 5.50978e+06（训练集 1853 样本，排除 head2） |

## 5. 方法

1. 冻结 FIND-Net（v2.1，epoch 58），只训 SSM
2. 两臂同预算、同早停
3. 评估两套：
   - **多轴对比** `compare_arms.py`：MAE/RMSE/PSNR/SSIM、p50/p90/p99/>400、Δhigh、Ez、面内 HF
   - **靶心判据** `v22_clinical_ab.py`：near/streak/far × 带限相关 5 尺度 + 边缘 + SSIM 三因子 + 自相关

## 6. 产物清单

| 路径 | 内容 |
|---|---|
| `results/tables/arm_compare.csv` | 两臂 × 11 项多轴指标 |
| `results/tables/clinical_ab.csv` | 两臂 × 三区 × 13 项靶心指标 |
| `results/figures/viz_exp008.png` | 三张代表片的四联对比（GT / 两臂 / 误差） |

本地镜像：`results/figures/`

## 7. 复现

```bash
# 训练（两臂同协议）
bash prism_framework/prism/scripts/mech_exp2.sh      # 含训练 + 多轴对比

# 靶心判据（两臂四度量）
python prism_framework/prism/scripts/v22_clinical_ab.py \
  --data-dir /root/autodl-tmp/PRISM/training_data_astra \
  --ckpts .../ssm_stage2_v22.pt,.../ssm_exp008_zfreq.pt \
  --names baseline,exp008_zfreq --max-seq 6 \
  --outdir /root/autodl-tmp/PRISM/v2.2/clinical_ab --device cuda

# 效果图
python prism_framework/prism/scripts/viz_arm_compare.py \
  --ckpts .../ssm_stage2_v22.pt,.../ssm_exp008_zfreq.pt \
  --names baseline,exp008_zfreq --n-slices 3 --out .../viz_exp008.png
```

源码快照：`src/`（`mech_exp2.sh`、`v22_clinical_ab.py`、`viz_arm_compare.py`）

## 8. 状态与已知问题

**结论摘要（3 行，分析见 `RESEARCH_LOG.md` → V2.2-baseline-SSM-2）**

1. holdout MAE：**36.9（存盘）/ 36.6（实际达到）** vs baseline 37.6 → 略好 1.9%~2.7%。
2. 多轴 11 项：**6 好 / 3 差**（MAE、p50、p90 变差；RMSE/PSNR/SSIM/p99/>400/Ez/HF 变好）。
3. 靶心：**near σ=1 相关 0.105 → 0.104 —— 没动**；21 项里赢 14，唯独没赢靶心。

**关键数字**

| | baseline | exp008 | |
|---|---|---|---|
| holdout MAE（最佳） | 37.6 | 36.9 / 36.6 | |
| near 带限相关 σ=1 | 0.1054 | **0.1042** | ❌ 没动 |
| near 带限相关 σ=16 | 0.2742 | 0.2969 | ✅ +8% |
| near 边缘幅值比 | 1.1535 | 0.9035 | ✅ 更接近 1 |
| near 方向余弦 | 0.3106 | 0.2861 | ❌ 更差 |
| near SSIM `l` | 0.5407 | 0.6596 | ✅ |
| near SSIM `c` / `s` | 0.8995 / 0.9080 | 0.9335 / 0.9231 | ✅ |
| 面内 HF | 0.7383 | 0.7980 | ✅ 更锐 |
| Ez | 0.8636 | 0.8988 | ✅ 过平滑减轻 |
| MAE | 36.9446 | 37.6734 | ❌ +2.0% |
| p90 | 87.7512 | 94.4197 | ❌ +7.6% |

**训练过程高频误差**（频域记录器，两步平均）

| step | exp008 | baseline |
|---|---|---|
| 199 | 0.0105 | 0.0093 |
| 399 | **0.0095** | 0.0100 |
| 599 | **0.0147** | — |

**已知问题**

| # | 问题 | 状态 |
|---|---|---|
| 1 | **早停判据（1% 最小改进）丢弃了更优的权重**：step 600 的 36.6 比存盘的 36.9 更好（0.8%），记为 bad 且未 save | ⬜ 应改为"严格更低即存盘" |
| 2 | `compare_arms` 的 `Δhigh` 是**份额**量，不可靠（见 FIND-NET-2 结论） | 已注明 |
| 3 | step 599 高频误差恶化（0.0147）提示**加权可能推过头** | 已记录，未拆解 |
| 4 | 两臂训练预算 800 步但实际都早停（500/700），步数不同 | 早停所致，非人为 |
| 5 | 单一 held-out 病人、6 条序列 | ⬜ |
| 6 | 效果图肉眼几乎无差别（exp008 略更花） | 已记录 |

## 9. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-11-22 | 训练完成（停 700，存盘 36.9） |
| 2026-11-22 | 多轴对比完成（6 好 3 差） |
| 2026-11-23 | 靶心判据完成（σ=1 0.105→0.104） |
| 2026-11-23 | 归档为 `V2.2-baseline-SSM-2` |
