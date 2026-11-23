# V2.2-baseline-FIND-NET-1

> **本文档的职责（三份文档分工）**
> - `../../RESEARCH_LOG.md` = **科研日志**：只写分析、推理、判断、对论文的含义
> - `experiments/<编号>/README.md` = **实验日志（本文）**：只写怎么做的、产物在哪、怎么复现、状态与问题
> - `../../DATA_LOG.md` = **数据日志**：只写数值、索引、口径、验证、作废记录

| 项 | 值 |
|---|---|
| 编号 | V2.2-baseline-FIND-NET-1（原 `exp012_prior_gain`） |
| 状态 | ✅ 定案 |
| 数据层 | 解析仿真（平行束 720 视角 / PSF σ=6.0） |
| 一句话 | FIND-Net 去掉的不只是金属伪影，还有重建算法伪影；其性能由**单切片可观测信息**决定 |

---

## 1. 对象与问题

| 部分 | 问题 |
|---|---|
| **P1** | 没有任何金属时，FIND-Net 还能优于"观测本身"吗？ |
| **P2** | 单切片上可用的信息变少时，它的性能怎么变？失效比例多少？ |

## 2. 环境

| 项 | 值 |
|---|---|
| 运行位置 | AutoDL 服务器 `connect.westb.seetacloud.com:13686` |
| 设备 | **CPU**（单切片前向 + radon/iradon） |
| conda env | **`agir`**（**必须**有 skimage；`pzneuro` 无 skimage） |

## 3. 配置

| 参数 | 值 |
|---|---|
| 权重 | `findnet_stage1_v2.1.pt`（No-GF，epoch 58） |
| FIND-Net 输入 | `(Baseline, LI, Mask)` —— stage1 原生格式 |
| 金属回填 | `dilate(Mask>0.5, r=8)` 处贴 Baseline |
| **评分域** | **`tissue_mask(mk, gt, r=8)`（三条曲线统一）** |
| P1 样本 | held-out `patients_head2`，`mask.max ≤ 1e-3`，按 `sample_id` 升序取 **40** 张 |
| P2 样本 | **155** 张含金属切片（金属量 0.6~472 当量） |

快照：`config/args.txt`

## 4. 数据

| 项 | 值 |
|---|---|
| 来源 | `/root/autodl-tmp/PRISM/training_data_astra/` |
| P1 | 无金属切片 → `Baseline` 即零金属 FBP |
| P2 | 含金属切片，覆盖 small/medium/large/giant |
| **P2 病人构成** | **head2 64 片 / head1 91 片**（head1 = 训练病人，in-sample） |

## 5. 方法

**P1**：对每张无金属切片，比较 `|Baseline−GT|` 与 `|FIND-Net−GT|`（同域）。

**P2**：三条曲线**全部在 `tissue_mask` 上**计算，按金属量分十分位取中位。

| 曲线 | 定义 |
|---|---|
| 植入前地板 | `simulate_sino(gt, 0, cat, seed)` → `iradon` → 对 GT 的误差 |
| Baseline | 含金属原始重建 |
| FIND-Net | 去伪影（含回填）后的输出 |

## 6. 产物清单

| 路径 | 内容 |
|---|---|
| `results/tables/findnet_decay.csv` | P2 逐片全量（155 行：metal / before / after / fn / d100 / f100_before / f100_after） |
| `results/figures/findnet_decay.png` | **P2 结论图**：三曲线（统一域）+ 分箱中位 |
| `results/figures/prior_gain_clean.png` | P1 五联：GT / 干净重建 / FIND-Net / 两张误差热力图 |
| `results/figures/prior_gain_scatter.png` | P1 逐片散点 |
| `results/figures/prior_gain_vs_metal.png` | ❌ 旧域 + 旧标题，**不用于结论** |

本地镜像：`results/figures/`、`results/figures/`

## 7. 复现

```bash
DATA=/root/autodl-tmp/PRISM/training_data_astra
V22=/root/autodl-tmp/PRISM/v2.2
AGIR=/root/miniconda3/envs/agir/bin/python

# P1
$AGIR -u prism_framework/prism/scripts/v22_prior_gain.py \
  --ckpt $DATA/findnet_stage1_v2.1.pt --patient patients_head2 --n-clean 40 \
  --outdir $V22/prior_gain --device cpu

# P2 第一步：155 片三状态（统一域）
$AGIR -u prism_framework/prism/scripts/v22_metal_sweep_tm.py \
  --data-dir $DATA --outdir $V22/metal_tm --nproc 8

# P2 第二步：三条曲线 + FIND-Net
$AGIR -u prism_framework/prism/scripts/v22_findnet_info_decay.py \
  --data-dir $DATA --sweep $V22/metal_tm/metal_sweep_tm.csv \
  --outdir $V22/findnet_decay --device cpu
```

源码快照：`src/`（三个脚本）

## 8. 状态与已知问题

**结论摘要（3 行，分析见 `RESEARCH_LOG.md` → V2.2-baseline-FIND-NET-1）**

1. P1：82.0 → 49.9 HU（1.64×，40/40）—— 去的是重建算法伪影。
2. P2：FN/地板 0.333 → 0.785，**146/155 = 94.2% 优于地板**。
3. 9 片失败：2 片重金属并列、7 片地板本身异常高（107~118 vs 中位 74.5）。

**已知问题**

| # | 问题 | 状态 |
|---|---|---|
| 1 | **155 片含 91 片训练病人** → P2 偏乐观 | ⬜ TODO T1 |
| 2 | P1 的 40 张非随机（升序取），且为相邻切片 | ⬜ 待抽样复核 |
| 3 | P1 输入 `Mask=0` 属 OOD | 已注明 |
| 4 | P1 逐片 MAE 未落盘（只输出均值与图） | ⬜ 待补 CSV |
| 5 | 图内 CJK 字形缺失 | ⬜ 待改英文标签 |
| 6 | P2 的 `fn` 用 LI 输入，与 -2/-3 的"纯 2D"（脏图自身）不一致 | 已注明，不可横向比 |

## 9. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-11-21 | 建立（`exp012_prior_gain`），P2 交叉点改为「不存在」 |
| 2026-11-22 | 改名 `V2.2-baseline-FIND-NET-1`；停用「先验增益」，改用「单切片可观测信息 $I_\sigma$」 |
| 2026-11-22 | **P2 按统一评分域重算**；「从不失效」修正为 **94.2%**；补 9 片失败解剖 |
| 2026-11-22 | **按三份文档分工重写为实验管理文档**（分析移至 RESEARCH_LOG） |
