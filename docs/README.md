# PRISM 文档与实验分析（docs/）

> 本目录是 **V2.2 baseline 分析**的完整记录：数学模型、实验数据、脚本快照、图表。
> 代码在 `prism/`、`analytic_sim/`；本目录只放**分析与其证据**。

---

## 0. 这是什么工作（一句话）

> **结构化测量缺失下的可恢复性分析框架** ——
> 把「哪里没有信息、还能恢复多少、该靠哪个通道补」形式化下来的工具。

表面上是 CT 金属伪影去除；实际上研究的是**部分信息逆问题**：
一个结构化的、可预测的测量缺失，摧毁了目标子空间里的大部分信息，
必须引入第二个零空间不与之重合的观测通道。

**CT 是这个框架的第一个落地实例，不是它的边界。**
完整抽象见 [`RESEARCH_LOG.md` §0](RESEARCH_LOG.md)。

---

## 1. 三份文档的分工

| 文档 | 只写 | 不写 |
|---|---|---|
| [`RESEARCH_LOG.md`](RESEARCH_LOG.md) | 分析、推理、判断、对论文的含义 | 配置/环境/命令/原始表 |
| [`DATA_LOG.md`](DATA_LOG.md) | 数值、索引、口径、验证、作废记录 | 为什么、意味着什么 |
| [`EXPERIMENTS_INDEX.md`](EXPERIMENTS_INDEX.md) | 实验目录与一句话结论 | — |
| `experiments/<编号>/README.md` | 怎么做的、产物在哪、怎么复现、状态与问题 | 长篇分析（3 行摘要 + 指向科研日志） |

---

## 2. 实验系列

### 一、原空间可观测性的量化（V2.2-baseline-FIND-NET-1 ~ -5）

把「信息在不在」变成一个可估计的泛函
$\rho_\sigma(r;m)=\operatorname{corr}(\mathrm{HP}_\sigma(y),\mathrm{HP}_\sigma(x))$，
并沿三个自变量（尺度 σ、退化程度 m、空间区域 r）测出来。

| 编号 | 回答 | 结果文件 |
|---|---|---|
| [-1](experiments/V2.2-baseline-FIND-NET-1/README.md) | 误差**多大**、随退化怎么变 | `findnet_decay.png/.csv` |
| [-2](experiments/V2.2-baseline-FIND-NET-2/README.md) | 误差**长什么样**（分布形状） | `err_dist_multi.png/.csv` |
| [-3](experiments/V2.2-baseline-FIND-NET-3/README.md) | 误差**在哪里**（空间定位） | `clinical_metrics.png/.csv`、`near_ring_*.png` |
| [-4](experiments/V2.2-baseline-FIND-NET-4/README.md) | 失败**为什么**（观测端天花板，不跑网络） | `observation_ceiling.csv` |
| [-5](experiments/V2.2-baseline-FIND-NET-5/README.md) | 「填充均值」假设的检验 | `mean_fill.png/.csv` |

### 二、SSM 机制诊断（V2.2-baseline-SSM-1 ~ -2）

| 编号 | 回答 | 结果文件 |
|---|---|---|
| [-1](experiments/V2.2-baseline-SSM-1/README.md) | 瓶颈在**递归**还是**出口** | `mu_structure.csv` |
| [-2](experiments/V2.2-baseline-SSM-2/README.md) | **频率加权**能否把结构找回来 | `arm_compare.csv`、`clinical_ab.csv`、`viz_exp008.png` |

---

## 3. 三个核心数字

| 量 | 值 | 含义 |
|---|---|---|
| $\rho_{2D}$（近环 σ=1） | **0.054** | 主通道在该子空间内几乎为零（$\rho^2$=0.3%） |
| $\rho_{z}$（z 上一片） | **0.280** | 第二通道保留三成（$\rho^2$=7.8%） |
| 现有递归取到 | **0.105** | 瓶颈在通道间传递，不在输出端 |

---

## 4. 复现

每个实验目录下：

```
README.md            怎么做的、产物在哪、怎么复现、已知问题
src/                 该实验的脚本快照
config/args.txt      复现命令
results/tables/      原始 CSV（权威数值）
results/figures/     PNG
```

环境：`pzneuro`（torch）或 `agir`（需 skimage）；GPU 12 GB 即可。
