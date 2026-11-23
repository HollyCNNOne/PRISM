# V2.2-baseline-SSM-1

> **本文档的职责（三份文档分工）**
> - `_scratch/RESEARCH_LOG.md` = **科研日志**：只写分析、推理、判断、对论文的含义
> - `experiments/<编号>/README.md` = **实验日志（本文）**：只写怎么做的、产物在哪、怎么复现、状态与问题
> - `_scratch/DATA_LOG.md` = **数据日志**：只写数值、索引、口径、验证、作废记录

| 项 | 值 |
|---|---|
| 编号 | V2.2-baseline-SSM-1 |
| 状态 | ✅ 定案 |
| 数据层 | 解析仿真（平行束 720 视角 / PSF σ=6.0） |
| 一句话 | 近环的细结构**没被 FIND-Net 抹掉，是递归根本没造出来**：先验 μ 只拿到 z 信息的 40%，出口再只丢 7% |

---

## 1. 对象与问题

> **瓶颈到底在哪一环：递归造不出结构，还是 FIND-Net 出口把它抹掉？**

这一个问题决定机制实验该往哪个方向做——改递归，还是改损失函数。

**做法**：把融合后的**先验 μ** 从管线里取出来，与 GT 求带限相关，
和 `out`（最终输出）、`obs`（观测）、`zprior`（z 上界）并排比。

| 落点 | 判定 |
|---|---|
| μ ≈ 0，out ≈ μ | 递归没造出来 → **改递归** |
| μ 明显 > 0，out ≪ μ | 出口抹掉了 → **改 loss** |
| μ ≈ zprior | 信息一直都在 → 改评估 |

## 2. 环境

| 项 | 值 |
|---|---|
| 运行位置 | AutoDL 服务器 `connect.westb.seetacloud.com:13686` |
| 设备 | GPU RTX 3080 Ti（12 GB） |
| conda env | `pzneuro`（torch） |

## 3. 配置

| 参数 | 值 |
|---|---|
| 检查点 | `ssm_stage2_v22.pt`（No-GF / nb=1 / kalman / bidirectional / max-z 8） |
| FIND-Net | `findnet_stage1_v2.1.pt`（No-GF，epoch 58） |
| 序列数 | 6（held-out `patients_head2` 最长的 6 条：Z=217×4, 178×2） |
| 区域 | near（8–30px 环带）/ streak（`\|LI\|>p90` 且非 near）/ far |
| 尺度 σ | 1 / 2 / 4 / 8 / 16 |
| 评分域 | `tissue_mask(mk, gt, r=8)` |

快照：`config/args.txt`

## 4. 数据

| 项 | 值 |
|---|---|
| 来源 | `/root/autodl-tmp/PRISM/training_data_astra/` |
| 病人 | 单一 held-out：`patients_head2` |
| 体素数 | near 36,650 / streak 2,047,063 / far 18,441,183 |

## 5. 方法

对每条序列跑一次完整双向递归，从 `res.fused` 取 `mu`：

```python
res = forward_backward(m, th, conf.d_ref, recursion=conf.recursion,
                       observer=obs, gain_mode=conf.gain_mode)
out = tr._final_output(res, dirty, masks, li_l)     # 最终输出
mu  = tr._stack([st.mu for st in res.fused])        # 融合先验
```

四个来源统一算 `corr(HP_σ(·), HP_σ(GT))`：

| 来源 | 含义 |
|---|---|
| **mu** | SSM 交给 FIND-Net 的融合先验 |
| **out** | 最终输出 x̂_z |
| obs | Baseline（观测本身） |
| zprior | GT 的上一片（z 方向 oracle 上界） |

## 6. 产物清单

| 路径 | 内容 |
|---|---|
| `results/tables/mu_structure.csv` | 三区 × 四来源 × 五尺度 |

本地镜像：`_scratch/images/V2.2/mu_structure/`

## 7. 复现

```bash
DATA=/root/autodl-tmp/PRISM/training_data_astra
python prism_framework/prism/scripts/v22_mu_structure.py \
  --data-dir $DATA --ckpt $DATA/ssm_stage2_v22.pt \
  --patient patients_head2 --max-seq 6 \
  --outdir /root/autodl-tmp/PRISM/v2.2/mu_structure --device cuda
```

约 35 分钟（6 条序列 × 完整递归）。源码快照：`src/v22_mu_structure.py`

## 8. 状态与已知问题

**结论摘要（3 行，分析见 `RESEARCH_LOG.md` → V2.2-baseline-SSM-1）**

1. near：μ 的 σ=1 = **0.113**，out = **0.105** → 出口只丢 **7%**，**不是出口的错**。
2. μ / zprior（捕获率）：σ=1 **40%**，σ=16 **60%** → **递归只拿到 z 的四到六成**。
3. out / μ（出口保留率）：σ=1 **93%**，σ=16 **67%** → **瓶颈随尺度互换**。

**已知问题**

| # | 问题 | 状态 |
|---|---|---|
| 1 | μ 是 `res.fused` 的融合后先验，**不是** M_μ 的原始预测 | 设计选择，已注明 |
| 2 | `zprior` 是 oracle（GT 上一片），非可实现上界 | 已注明 |
| 3 | 单一病人、6 条序列 | ⬜ |
| 4 | near 仅 36,650 体素（组织体素 0.18%） | ⬜ |
| 5 | `streak` 定义抓到骨缘（TODO T4） | ⬜ |
| 6 | 无图（表即证据） | 可接受 |

## 9. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-11-23 | 建立并定案 |
