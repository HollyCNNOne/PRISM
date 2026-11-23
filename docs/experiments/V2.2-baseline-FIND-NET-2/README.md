# V2.2-baseline-FIND-NET-2

> **本文档的职责（三份文档分工）**
> - `docs/RESEARCH_LOG.md` = **科研日志**：只写分析、推理、判断、对论文的含义
> - `docs/experiments/<编号>/README.md` = **实验日志（本文）**：只写怎么做的、产物在哪、怎么复现、状态与问题
> - `docs/DATA_LOG.md` = **数据日志**：只写数值、索引、口径、验证、作废记录

| 项 | 值 |
|---|---|
| 编号 | V2.2-baseline-FIND-NET-2（原 `exp009 §5.4`） |
| 状态 | ✅ 定案 |
| 数据层 | 解析仿真（平行束 720 视角 / PSF σ=6.0） |
| 一句话 | FIND-Net 的误差是**尖峰重尾**（不是弥散模糊）：中位数更好，p99 在 large/giant 反超 |

---

## 1. 对象与问题

> 把误差从"一个平均值"拆成"一条分布"：这堆误差是"大家都错一点"，还是"多数很好、少数错得离谱"？

## 2. 环境

| 项 | 值 |
|---|---|
| 运行位置 | AutoDL 服务器 `connect.westb.seetacloud.com:13686` |
| 设备 | **CPU**（12 次 FIND-Net 前向 + 12 次 radon/iradon） |
| conda env | **`agir`**（需要 skimage） |

## 3. 配置

| 参数 | 值 |
|---|---|
| 权重 | `findnet_stage1_v2.1.pt`（epoch 58） |
| FIND-Net 输入 | `(Baseline, LI, Mask)` |
| 病人 | `patients_head2`（held-out） |
| **代表片取法** | **每条序列中金属量最大的切片** |
| 序列数 | **12**（每 size 3 条） |
| 分箱 | `<50 / 50-100 / 100-200 / 200-400 / >400` HU |
| 评分域 | `tissue_mask(mk, gt, r=8)` |

快照：`config/args.txt`

## 4. 数据

held-out `patients_head2` 共 12 条序列，每 size 3 条。代表片金属量：

| size | 三条序列的代表片金属量 |
|---|---|
| small | 62 / 58 / 51 |
| medium | 293 / 294 / 205 |
| large | 2331 / 724 / 652 |
| giant | 2245 / 1917 / 2110 |

> **为什么取金属量最大的那一片**：线圈是三维体，沿 z 的横截面是「小 → 标称尺寸 → 小」。
> 其余切片只切到线圈边缘，会系统性低估该植入体的 size。

## 5. 方法

三个状态，同一评分域：

| 状态 | 定义 |
|---|---|
| **before** | `simulate_sino(gt, 0, cat, seed)` → `iradon`（零金属同管线） |
| **after** | Baseline |
| **FN** | FIND-Net + 金属回填 |

每状态在该切片组织体素上取 `p50/p90/p99/max` + 五分箱占比；**跨代表片取中位**（每片贡献相同）。

## 6. 产物清单

| 路径 | 内容 |
|---|---|
| `results/tables/err_dist_multi.csv` | 逐片逐状态全量（12 片 × 3 状态） |
| `results/figures/err_dist_multi.png` | 四尺寸 × 三状态 CDF（跨代表片平均） |

本地镜像：`results/figures/`

## 7. 复现

```bash
DATA=/root/autodl-tmp/PRISM/training_data_astra
OMP_NUM_THREADS=4 /root/miniconda3/envs/agir/bin/python -u \
  prism_framework/prism/scripts/v22_error_dist_multi.py \
  --data-dir $DATA --patient patients_head2 \
  --outdir /root/autodl-tmp/PRISM/v2.2/err_dist_multi --device cpu
```

源码快照：`src/v22_error_dist_multi.py`

## 8. 状态与已知问题

**结论摘要（3 行，分析见 `RESEARCH_LOG.md` → V2.2-baseline-FIND-NET-2）**

1. FN/before 的 p50 比：0.340 / 0.421 / 0.649 / 0.687（四档全部更好）。
2. p99 比：0.699 / 0.604 / **1.166 / 1.491** —— 分界点在 **medium 与 large 之间**。
3. >400HU 在 large/giant 反超地板：2.47% vs 1.94%；3.09% vs 1.60%。

**已知问题**

| # | 问题 | 状态 |
|---|---|---|
| 1 | 每 size 仅 3 张代表片、单一病人 | ⬜ TODO T3（生成更多 head2 植入体） |
| 2 | 分位数是**片内体素级**，不是跨病例分布 | 已注明 |
| 3 | 代表片只反映"标称尺寸"，不反映线圈边缘切片 | 设计选择，已注明 |
| 4 | 未含大错的空间定位（哪些体素） | 由 -3 部分回答 |
| 5 | 未加 GT 图像面板 | ⬜ TODO T2 |

## 9. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-11-21 | 初版（`exp009 §5.4`），**每尺寸仅 1 张切片** |
| 2026-11-22 | **重建为 V2.2-baseline-FIND-NET-2**：每序列取代表片，12 条 held-out（每 size 3 张）；旧单例版作废 |
| 2026-11-22 | 按三份文档分工重写为实验管理文档 |
