# 实验索引（V2.2 baseline 阶段）

> **三份文档的分工**
> - `RESEARCH_LOG.md` = **科研日志**：只写分析、推理、判断、对论文的含义
> - `DATA_LOG.md` = **数据日志**：只写数值、索引、口径、验证、作废记录
> - `experiments/<编号>/README.md` = **实验日志**：只写怎么做的、产物在哪、怎么复现、状态与问题
>
> 本仓库只收录 **V2.2 baseline 阶段**（FIND-NET-1~5 度量 + SSM-1~3 诊断）。
> 更早的阶段记录保留在本地工作区，未纳入本仓。


---

## 目录

```
experiments/<编号>/
├── README.md          9 节：对象/环境/配置/数据/方法/产物/复现/状态/变更
├── src/               脚本快照
├── config/args.txt    复现命令
└── results/
    ├── tables/        原始 CSV（权威数值）
    └── figures/       PNG
```

## 一、原空间可观测性的量化（FIND-NET-1 ~ -5）

| 编号 | 回答 | 一句话结论 |
|---|---|---|
| [V2.2-baseline-FIND-NET-1](experiments/V2.2-baseline-FIND-NET-1/README.md) | 误差**多大**：去掉哪两类伪影 + 性能由什么决定 | 无金属 40 片 82.0→49.9（1.64×）；FN/地板 0.333→0.785；**94.2% 优于干净重建** |
| [V2.2-baseline-FIND-NET-2](experiments/V2.2-baseline-FIND-NET-2/README.md) | 误差**长什么样**：分布形状 | **尖峰重尾**：p50 比 0.34~0.69 全程更好，**p99 在 large/giant 反超 1.17 / 1.49** |
| [V2.2-baseline-FIND-NET-3](experiments/V2.2-baseline-FIND-NET-3/README.md) | 误差**在哪里**：空间定位 | **金属外 8–30px 环带**：带限相关 0.100、边缘 3.48×、方向余弦 0.29、SSIM l 0.44 |
| [V2.2-baseline-FIND-NET-4](experiments/V2.2-baseline-FIND-NET-4/README.md) | 失败**为什么**：观测端天花板（不跑网络） | 观测高频相关 **0.054**、携带 **99.34×** 能量；streak 0.824 / far 0.427–0.941 |
| [V2.2-baseline-FIND-NET-5](experiments/V2.2-baseline-FIND-NET-5/README.md) | 「填充均值」假设的检验 | **否证**：FIND-Net 185.1 vs μ_local 34.7（差 5.3×）——是加伪影，不是抹平 |

## 二、SSM 机制诊断（SSM-1 ~ -3）

| 编号 | 回答 | 一句话结论 |
|---|---|---|
| [V2.2-baseline-SSM-1](experiments/V2.2-baseline-SSM-1/README.md) | 瓶颈在**递归**还是**出口** | μ 捕获 z 的 **40%**，出口只再丢 7% → **瓶颈在递归** |
| [V2.2-baseline-SSM-2](experiments/V2.2-baseline-SSM-2/README.md) | **频率加权**能否把结构找回来 | 赢 14/21 项，**唯独 near σ=1 未动**（0.1054→0.1042）；MAE 变差 2.0% |
| [V2.2-baseline-SSM-3](experiments/V2.2-baseline-SSM-3/README.md) | 细节传递是否随**到信息片距离**衰减 | **每片 ×0.60（Δ≈0.40）**；σ=16 每片 ×0.85 → 金属跨 10 片时细节仅剩 8% |

---

## 一条线读下来

1. **2D 在近环没有信息**（FIND-NET-4：相关 0.054、99.34× 假能量）
2. **失败是「假」不是「糊」**（FIND-NET-3），**且集中在贴金属的 8–30px**
3. **不是"填充均值"**（FIND-NET-5：比均值填充还差 5.3×）
4. **唯一剩下的通道是 z**（0.280，但只有 z-稳定结构）
5. **现有递归只取到 40%**（SSM-1），**因为传递每片 ×0.6**（SSM-3）
6. **改损失函数无效**（SSM-2：赢 14/21 项，唯独输掉靶心）

> ⇒ 下一刀必须砍在**递归的状态传递**，不是损失函数。

