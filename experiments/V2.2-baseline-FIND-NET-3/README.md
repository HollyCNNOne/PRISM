# V2.2-baseline-FIND-NET-3

> **本文档的职责（三份文档分工）**
> - `../../RESEARCH_LOG.md` = **科研日志**：只写分析、推理、判断、对论文的含义
> - `experiments/<编号>/README.md` = **实验日志（本文）**：只写怎么做的、产物在哪、怎么复现、状态与问题
> - `../../DATA_LOG.md` = **数据日志**：只写数值、索引、口径、验证、作废记录

| 项 | 值 |
|---|---|
| 编号 | V2.2-baseline-FIND-NET-3（原 `exp014_near_ring_region`） |
| 状态 | ✅ 定案 |
| 数据层 | 解析仿真（平行束 720 视角 / PSF σ=6.0） |
| 一句话 | 误差被精确定位在**金属外 8–30px 环带**；该区失败形态是「假」不是「糊」 |

---

## 1. 对象与问题

| 问 | 内容 |
|---|---|
| 位置 | 误差集中在哪个区域？ |
| 形态 | 那个区域的输出，是"糊"还是"假"？ |
| 因果 | 那个区域为什么失败——模型问题，还是那里的输入本身坏了？ |

## 2. 环境

| 项 | 值 |
|---|---|
| 运行位置 | AutoDL 服务器 `connect.westb.seetacloud.com:13686` |
| 设备 | GPU RTX 4090D |
| conda env | `pzneuro`（torch） |

## 3. 配置

| 参数 | 值 |
|---|---|
| 检查点 | `ssm_stage2_v22.pt`（No-GF / nb=1 / kalman / bidirectional / max-z 8） |
| FIND-Net | `findnet_stage1_v2.1.pt`（No-GF，epoch 58） |
| 序列数 | 6（held-out `patients_head2` 最长的 6 条：Z=217×4, 178×2） |
| 近环内径 / 外径 | **8 px / 30 px** |
| 尺度 σ | 1 / 2 / 4 / 8 / 16 |
| 边缘阈值 | `\|∇GT\| > p90`（分区内） |
| 评分域 | `tissue_mask(mk, gt, r=8)` |

快照：`config/args.txt`

### 区域定义（只用推理期可见的输入，不用 GT）

| 区域 | 定义 | 体素数 |
|---|---|---|
| **near** | `dilate(mask>1e-3, 8px)` → `dilate(..., 30px)` 环带 ∩ 评分域 | **36,650** |
| streak | `\|LI\| > p90`（评分域内）且不属于 near | 2,047,063 |
| far | 评分域 − near − streak | 18,441,183 |

## 4. 数据

| 项 | 值 |
|---|---|
| 来源 | `/root/autodl-tmp/PRISM/training_data_astra/` |
| 病人 | 单一 held-out：`patients_head2` |
| 体素占比 | near 占组织体素 **0.18%**（36,650 / 20,524,896），约单切片 1.2% |

## 5. 方法

**两条模型臂**

| 臂 | 定义 |
|---|---|
| `findnet` | 纯 2D：观测闭包冷启动调用一次（μ_pred = 脏图），无递归先验 |
| `ssm` | 完整管线（融合先验 + 金属回填） |

**四把与 MAE 正交的尺子**

| # | 度量 | 定义 |
|---|---|---|
| ① | 多尺度带限相关 | `corr(HP_σ(OUT), HP_σ(GT))`，`HP_σ(x)=x−Gaussian(x,σ)` |
| ② | 边缘保持率 | GT 强边缘处 `mean\|∇OUT\|/mean\|∇GT\|` + 梯度方向余弦 |
| ③ | SSIM 三因子 | `l`（亮度）· `c`（对比）· `s`（结构） |
| ④ | 误差空间自相关 | 误差图 lag=1 / lag=8 的相关系数 |

## 6. 产物清单

| 路径 | 内容 |
|---|---|
| `results/tables/clinical_metrics.csv` | 三区 × 两臂 × 四度量全量 |
| `results/figures/clinical_metrics.png` | ① 带限相关 vs 尺度 ／ ② 边缘保持 ／ ③ SSIM 结构项 |
| `results/figures/near_ring_viz.png` | 整片四联 + 误差行（青线=环带边界，红线=金属核） |
| `results/figures/near_ring_zoom.png` | 金属局部放大（条纹 → 减轻 → 抹平成均匀灰） |

本地镜像：`results/figures/`

## 7. 复现

```bash
DATA=/root/autodl-tmp/PRISM/training_data_astra
V22=/root/autodl-tmp/PRISM/v2.2

# 四度量
python prism_framework/prism/scripts/v22_clinical_usability.py \
  --data-dir $DATA --ckpt $DATA/ssm_stage2_v22.pt \
  --patient patients_head2 --max-seq 6 --outdir $V22/clinical --device cuda

# 近环示意图 + 局部放大
python prism_framework/prism/scripts/viz_near_ring.py \
  --data-dir $DATA --ckpt $DATA/ssm_stage2_v22.pt \
  --patient patients_head2 --out $V22/clinical/near_ring_viz.png --device cuda
```

源码快照：`src/`（`v22_clinical_usability.py`、`viz_near_ring.py`）

## 8. 状态与已知问题

**结论摘要（3 行，分析见 `RESEARCH_LOG.md` → V2.2-baseline-FIND-NET-3）**

1. 只有 near 塌：带限相关 σ=1 **0.100**（far 0.803 / streak 0.844），σ=16 仍只有 0.265。
2. 形态是「假」：梯度 **3.480×**、方向余弦 **0.292**、SSIM `l` **0.442**。
3. 失败位置 = 观测被摧毁的位置（该区观测相关 **0.054**、**99.34×** 假能量）。

**已知问题**

| # | 问题 | 状态 |
|---|---|---|
| 1 | `streak` 用 `\|LI\|>p90` 抓到**骨缘**而非条纹 | ⬜ TODO T4 |
| 2 | near 外径 30px 未做敏感性扫描 | ⬜ |
| 3 | near 仅 36,650 体素、单一病人 | ⬜ |
| 4 | 区域是**几何定义**，未与物理量 `d_j` 对齐 | ⬜ |
| 5 | 无 Baseline 列的带限相关（"far 好"只是相对 GT） | ⬜ |
| 6 | 残留条纹的**形状**未测（指标对方向不敏感） | ⬜ **TODO T5（重点）** |
| 7 | 本臂的 LIct 用脏图自身，与 -1 用 LI 不一致 | 已注明，不可横向比 |

## 9. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-11-22 | 建立（原名 `exp014_clinical_usability`，范围较宽） |
| 2026-11-22 | 重新定位为「近环区域考察」→ 目录 `exp014_near_ring_region` |
| 2026-11-22 | 改名 `V2.2-baseline-FIND-NET-3`，定位收敛为「失败在哪里」 |
| 2026-11-22 | 按三份文档分工重写为实验管理文档（分析移至 RESEARCH_LOG） |
