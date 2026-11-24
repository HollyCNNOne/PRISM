# PRISM —— 物理引导的递归推理金属伪影去除

颅内铂钨（Pt-W）弹簧圈栓塞随访 CT 的金属伪影去除（MAR）。

本仓当前只含**解析仿真线路 `analytic_sim/`**：用 skimage 平行束（radon/iradon）自包含地
生成成对 MAR 数据（Baseline / LI / d_j / g），是 PRISM 递归状态空间模型（SSM）训练侧
「解析域」的唯一数据来源。全物理（CatSim/XCIST）线路在另一条线，不在此仓。

---

## analytic_sim/ —— 解析仿真线路（V4 定案）

### 物理管线（`parallel_beam.py` 单一真源）

```
L_tissue = radon((1-mask)·hu2mu(clean))·PIXEL_MM   组织线积分
L_metal  = radon(mask·METAL_MU)·PIXEL_MM           金属线积分（μ=7.02 mm⁻¹ = 350000 HU）
L_tissue ← L_tissue·(1 − β_bh·L_tissue)            束硬化杯突（仅组织）
L = L_tissue + L_metal
I = I0·exp(−L)
I_noisy = poisson(I) + N(0, σ_e)                   光子 + 电子噪声地板
I_noisy = max(I_noisy, 0.5)
L_noisy = −log(I_noisy/I0)
L_noisy ← L_noisy + N(0, σ_starve)·1{L > L_starve} 光子饥饿阴影区 L 域噪声
L_noisy ← gaussian_filter1d(L_noisy, σ=PSF[size])  尺寸相关角度轴 PSF 模糊
```

| 常量 | 值 | 说明 |
| --- | --- | --- |
| PSF_BY_SIZE | small 1.5 / medium 2.0 / large 4.5 / giant 6.0 | 角度轴（axis=1）高斯模糊，最接近临床「脏、糊」星芒 |
| σ_e | 20 | 电子噪声地板（counts），定阴影暗度 |
| β_bh | 1.5e-2 | 束硬化系数（仅组织）→ 杯突 |
| σ_starve | 3.0 | 光子饥饿区（L>6）L 域噪声 → 斑驳暗带 |
| I0 | 1e5 | 入射光子数 |
| METAL_MU | 7.02 mm⁻¹ | Pt-W 金属衰减 = 350000 HU |

### 文件

| 文件 | 职责 |
| --- | --- |
| `parallel_beam.py` | 单一物理真源 `simulate_sino(clean, mask, cat, seed)`。同 seed、同 rng 顺序，保证 Baseline/LI/d_j/g 共享同一噪声实现 |
| `regen_skimage.py` | 重生成 `train/Baseline` + `train/LI`（线性插值 MAR + shepp-logan FBP + 金属回填） |
| `regen_phys.py` | 重生成 `train/Phys/{d_j,g}`：`d_j = AᵀW(A·1)`，`w = I0·exp(−L_noisy)`，`g = 1/(1+(d_j/d_ref)³)` |
| `render_analytic_mar.py` | 探针：4 尺寸 Target/LI/Baseline 网格渲染 |
| `render_z_continuity.py` | 探针：单序列 z 轴连续渲染（验证 SSM 递归先验的 z 连续性） |

### 用法（服务器 agir env）

```bash
/root/miniconda3/envs/agir/bin/python -u regen_skimage.py --nproc 16
/root/miniconda3/envs/agir/bin/python -u regen_phys.py --nproc 16
```

输入 `train/Target/{id}.raw`（干净 HU）+ `train/Mask/{id}.raw`（金属体积分数），
覆盖输出 `train/Baseline`、`train/LI`、`train/Phys/{id}_{d_j,g}.npy`。
随机种子 = `int(sample_id 后缀)`，nproc 顺序无关（确定性）。

### 为什么 d_j/g 用平行束重生成

d_j（数据充分度）与 g（信任门控）是 Stage 2 SSM 观测精度 `d_j/d_ref` 与阴影门
`s=1{r≥r0}` 的物理量，必须与 Baseline 重建**同几何**。旧版扇束 ASTRA（strip_fanflat 1000
视角）→ 新版 skimage 平行束（720 视角），d_j 空间图案（金属背后阴影走向）随几何变。
`d_j/d_ref` 是 scale-invariant（d_ref=训练集冻结 median 重新锚定），故平行束绝对量级无关，
只要空间图案对。`regen_phys.py` 会自动删陈旧 `d_ref_*.npy` 缓存。

### 安全边界

本仓只含**解析管线**。全物理（CatSim/XCIST：`batch_simulate.py` / `generate_coil_3d_mask.py`
/ `generate_training_data.py`）在另一条线，**不在此仓**，请勿在此混入或修改全物理代码。

---

---

## 三份文档（V2.2 baseline 阶段）

本仓只收录 **V2.2 baseline 阶段**：FIND-NET-1~5（度量）+ SSM-1~3（诊断）。
更早阶段的记录保留在本地工作区，不在此仓。

| 文档 | 内容 | 规模 |
|---|---|---|
| [`RESEARCH_LOG.md`](RESEARCH_LOG.md) | **科研日志**：数学抽象、推理、判断、对论文的含义 | 11 节 |
| [`DATA_LOG.md`](DATA_LOG.md) | **数据日志**：数值、口径、验证、作废、产物索引 | 8 节 |
| [`EXPERIMENTS_INDEX.md`](EXPERIMENTS_INDEX.md) | **实验索引**：8 个实验的入口与一句话结论 | — |
| [`experiments/<编号>/`](experiments/) | **实验日志**：每个实验的 README + src + config + results | 8 个 |

### 这一阶段在做什么

把「金属伪影去除」抽象为「**结构化测量缺失下的可恢复性分析**」：

- 测量算子存在**结构化零空间**（Hanson 1982 / Louis 1986）
- 金属不制造"噪声"，而是制造一个**可预测的信息缺失子空间**
- 于是问题不是"去噪"，是"**度量还能恢复多少、靠哪个通道补**"

### 核心数字（近环，σ=1）

| 量 | 值 | 含义 |
|---|---|---|
| ρ_2D | **0.054** | 主通道在目标子空间内几乎为零（ρ²=0.3%，99.7% 不可解释） |
| ρ_z | **0.280** | 第二通道（相邻切片）保留三成，且只覆盖 z-稳定结构 |
| 现管线 | **0.105** | 捕获 z 的 40% |
| 递归传递系数 | **0.60/片** | 金属跨 10 片时，最深处细节仅剩 8% |

### 一条线读下来

1. 2D 在近环**没有信息**（FIND-NET-4）
2. 失败是「**假**」不是「糊」，且集中在金属外 8–30px（FIND-NET-3）
3. 不是"填充均值"——**比均值填充还差 5.3 倍**（FIND-NET-5）
4. 唯一剩下的通道是 **z**（FIND-NET-4）
5. 现有递归只取到 40%，**因为传递每片 ×0.6**（SSM-1 / SSM-3）
6. **改损失函数无效**（SSM-2：赢 14/21 项，唯独输掉靶心）

> ⇒ 下一刀必须砍在**递归的状态传递**，不是损失函数。

---
