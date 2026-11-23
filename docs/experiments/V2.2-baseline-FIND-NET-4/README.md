# V2.2-baseline-FIND-NET-4

> **本文档的职责（三份文档分工）**
> - `docs/RESEARCH_LOG.md` = **科研日志**：只写分析、推理、判断、对论文的含义
> - `docs/experiments/<编号>/README.md` = **实验日志（本文）**：只写怎么做的、产物在哪、怎么复现、状态与问题
> - `docs/DATA_LOG.md` = **数据日志**：只写数值、索引、口径、验证、作废记录

| 项 | 值 |
|---|---|
| 编号 | V2.2-baseline-FIND-NET-4 |
| 状态 | ✅ 定案 |
| 数据层 | 解析仿真（平行束 720 视角 / PSF σ=6.0） |
| 一句话 | **不跑网络**，直接测观测端还剩多少高频信息：近环 0.054 / 99.34× 假能量 |

---

## 1. 对象与问题

> 近环的失败，是**模型不够好**，还是**数据里本来就没有**？

这个区分必须做，因为两者的破解方式完全不同，而"模型不够好"是一个永远可以提、永远不用负责的质疑。

**设计目标：让这个区分不依赖我们的模型。**

## 2. 环境

| 项 | 值 |
|---|---|
| 运行位置 | AutoDL 服务器 `connect.westb.seetacloud.com:13686` |
| 设备 | **CPU（纯 numpy/scipy，不需要 torch）** |
| conda env | `pzneuro`（只用 numpy/scipy） |

## 3. 配置

| 参数 | 值 |
|---|---|
| 序列 | held-out `patients_head2` 最长的 6 条（与 -3 完全相同） |
| 尺度 σ | 1 / 2 / 4 / 8 / 16 |
| 高通算子 | `HP_σ(x) = x − Gaussian(x, σ)` |
| 区域 | near（8–30px 环带）/ streak（`\|LI\|>p90` 且非 near）/ far |
| 表面高频能量 | `x − Gaussian(x,3)` 的均方 |
| 评分域 | `tissue_mask(mk, gt, r=8)` |

快照：`config/args.txt`

## 4. 数据

| 项 | 值 |
|---|---|
| 来源 | `/root/autodl-tmp/PRISM/training_data_astra/` |
| 体素数 | near 36,650 / streak 2,047,063 / far 18,441,183 |
| 病人 | 单一 held-out：`patients_head2` |

## 5. 方法

对每个区域、每个 σ，计算三个来源与真值高频的相关：

```
corr( HP_σ(Baseline),        HP_σ(GT) )   # 观测本身
corr( HP_σ(Gaussian(BL,4)),  HP_σ(GT) )   # 观测能做到的最好低通
corr( HP_σ(GT 的上一片),      HP_σ(GT) )   # z 方向（对照）
```

另输出各区域 GT 与 Baseline 的面内高频能量绝对值。

**语义**：该相关 = 以观测高频为唯一输入的**最优线性估计器**（含 Wiener）对真值高频的解释力。
深度网络可利用非线性结构**略超**它（实测 FIND-Net 0.100 > 观测 0.054），故它是**线性天花板**，非绝对上界。

## 6. 产物清单

| 路径 | 内容 |
|---|---|
| `results/tables/observation_ceiling.csv` | 三区 × 三来源 × 五尺度 + 各区 GT/Baseline 高频能量 |

本地镜像：`results/figures/`

## 7. 复现

```bash
python prism_framework/prism/scripts/v22_observation_ceiling.py \
  --data-dir /root/autodl-tmp/PRISM/training_data_astra \
  --patient patients_head2 --max-seq 6 \
  --outdir /root/autodl-tmp/PRISM/v2.2/ceiling
```

纯 CPU，约 5 分钟。源码快照：`src/v22_observation_ceiling.py`

## 8. 状态与已知问题

**结论摘要（3 行，分析见 `RESEARCH_LOG.md` → F6）**

1. 近环观测高频与真值相关 **0.054**（σ=1）→ 0.159（σ=16），却携带 **99.34×** 真值的能量。
2. 对照组正常：streak 0.824 / far 0.427→0.941。
3. z 上一片 0.280（σ=1）= 观测的 **5.2 倍**。

**已知问题**

| # | 问题 | 状态 |
|---|---|---|
| 1 | 0.054 是**线性**天花板；非线性上界（条件互信息）未算 | ⬜ |
| 2 | `zprior` 用 GT 的上一片，是 **oracle** 上界 | 已注明 |
| 3 | 区域沿用 -3 的定义（`streak` 抓到骨缘，TODO T4） | ⬜ |
| 4 | 单一病人、6 条序列 | ⬜ |
| 5 | 无图（表即证据） | 可接受 |

## 9. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-11-22 | 建立并定案 |
| 2026-11-22 | 按三份文档分工重写为实验管理文档；标注 F6 编号 |
