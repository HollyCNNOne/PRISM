# V2.2-baseline-FIND-NET-5

> **本文档的职责（三份文档分工）**
> - `docs/RESEARCH_LOG.md` = **科研日志**：只写分析、推理、判断、对论文的含义
> - `docs/experiments/<编号>/README.md` = **实验日志（本文）**：只写怎么做的、产物在哪、怎么复现、状态与问题
> - `docs/DATA_LOG.md` = **数据日志**：只写数值、索引、口径、验证、作废记录

| 项 | 值 |
|---|---|
| 编号 | V2.2-baseline-FIND-NET-5 |
| 状态 | ✅ 定案 |
| 数据层 | 解析仿真（平行束 720 视角 / PSF σ=6.0） |
| 一句话 | 检验「FIND-Net 在信息缺失处填充均值」——**否证**：它比均值填充还差 5.3 倍 |

---

## 1. 对象与问题

> 信息缺失处，网络输出退化成什么？是"局部均值"（放弃结构、只留趋势），还是别的？

若"填充均值"成立，含义很重：说明 2D 先验已到极限，且极限就是"没有细节"。

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
| 序列 | held-out `patients_head2` 最长的 6 条（Z=217×4, 178×2） |
| 近环内径 / 外径 | 8 px / 30 px |
| μ_local 尺度 | **σ = 8** |
| 条纹阈值 | `\|LI\| > p90` |
| 评分域 | `tissue_mask(mk, gt, r=8)` |

快照：`config/args.txt`

## 4. 数据

| 项 | 值 |
|---|---|
| 来源 | `/root/autodl-tmp/PRISM/training_data_astra/` |
| 体素数 | near 36,650 / streak 2,047,063 / far 18,441,183 |
| 病人 | 单一 held-out：`patients_head2` |

## 5. 方法

### 5.1 判据形式化

若输出 = 局部均值，则 `MAE(FIND-Net) ≈ MAE(μ_local)`，其中 `μ_local = Gaussian(GT, σ=8)`
代表「2D 先验的贝叶斯极限（零信息解）」。

### 5.2 七个参照

| 参照 | 含义 |
|---|---|
| `baseline` | 原始重建（观测本身） |
| `mu_local` | `Gaussian(GT, σ=8)` |
| `findnet` | 纯 2D（观测闭包冷启动调用一次） |
| `ssm` | 完整管线 |
| `zprior` | GT 的上一片（z 方向上界） |
| `_blur_obs` | `Gaussian(Baseline, σ=4)`（观测能做到的最好低通） |
| `_gt_hf` | GT 自身（给出该区高频能量的绝对量级，用于判断相关是否被噪声主导） |

### 5.3 指标

`MAE` ｜ `面内高频能量 (x−Gaussian_σ3(x))²` ｜ `corr(·, μ_local)` ｜ `corr(·, GT)`

## 6. 产物清单

| 路径 | 内容 |
|---|---|
| `results/tables/mean_fill.csv` | 三区 × 七参照 × 四指标（21 行） |
| `results/figures/mean_fill.png` | 左：绝对 MAE；右：相对 μ_local 的比值 |

本地镜像：`results/figures/`

## 7. 复现

```bash
DATA=/root/autodl-tmp/PRISM/training_data_astra
python prism_framework/prism/scripts/v22_mean_fill_test.py \
  --data-dir $DATA --ckpt $DATA/ssm_stage2_v22.pt \
  --patient patients_head2 --max-seq 6 \
  --outdir /root/autodl-tmp/PRISM/v2.2/mean_fill --device cuda
```

冒烟建议：`--max-seq 1` 先跑通（约 10 分钟）。源码快照：`src/v22_mean_fill_test.py`

## 8. 状态与已知问题

**结论摘要（3 行，分析见 `RESEARCH_LOG.md` → F7）**

1. near：FIND-Net MAE **185.1** vs μ_local **34.7** → **5.33×**；HF 能量 36,418 vs 21.9 → **1663×**。
2. 方向是反的：它比均值填充更**粗糙**，不是更平滑。
3. SSM 把 HF 能量压到 **2,946**（GT 2,850 的 1.03×），但 MAE 仍是 μ_local 的 **2.04×**。

**已知问题**

| # | 问题 | 状态 |
|---|---|---|
| 1 | **`μ_local` 不能当判决标准**：streak 区给出 512.0 HU 的荒谬误差（它是故意糊掉边缘的 oracle） | 已注明，仅作参照之一 |
| 2 | `zprior` 是 oracle（GT 上一片） | 已注明 |
| 3 | near 仅 36,650 体素、单一病人 | ⬜ |
| 4 | `streak` 定义抓到骨缘（TODO T4） | ⬜ |
| 5 | `corr(·,GT)` 是未高通的相关，由低频主导，判别力弱 | 辅助指标 |
| 6 | 第一版曾因 `μ_local` 误作判据、缺 baseline 参照、corr 未打印、键缺失崩溃 | 已修 |

## 9. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-11-22 | v1：`μ_local` 误作判据、缺 baseline、corr 未打印、`_blur_obs` 键缺失崩溃 |
| 2026-11-22 | v2：加 `baseline`/`_blur_obs`/`_gt_hf`，打印全部相关性，绘图包 try/except；定案 |
| 2026-11-22 | 按三份文档分工重写为实验管理文档；标注 F7 编号 |
