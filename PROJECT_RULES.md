# PROJECT_RULES.md —— PRISM v2.x 原子化版本控制纪律

> 本文件是 v2.x 阶段（增量优化）的最高工程规范。所有代码修改必须严格逐条遵守。

---

## 1. 分支隔离与底座保护

- 动作：以当前 `main` 分支（或 `v1.3-baseline`）为基础，创建并切换到全新开发分支：
  `git checkout -b feature/v2_incremental`
- 铁律：所有 v2.x 的代码修改，**绝对不允许直接在 main 分支上进行**。`main` 分支必须永远保持 v1.3 的纯净与可运行状态。

## 2. 单点修改与验证（严格禁止"一锅乱炖"）

- 动作：每次只允许针对**一个方向**的代码进行修改（例如：做 Mamba 改造时，绝不允许同时去碰 FFL Loss）。
- 验证门槛：修改完一个点后，必须先在终端**跑通 50 步单 Batch Overfit 测试**，并保存对比图。

## 3. 日志先行（No Log, No Commit）

- 动作：在验证通过后、执行 `git commit` 之前，**必须先修改 EXPERIMENT_LOG.md**。
- 日志内容：必须写明本次改动的【目的】、【修改的具体文件与行数】、【Overfit 验证结果的指标/视觉变化】。

## 4. 规范的 Commit 与打 Tag

- 动作：仅 `git add` 被修改的核心代码文件（**严禁 `git add .`** 导致垃圾文件混入）。
- Commit 规范：必须使用规范化前缀。例如：
  - `feat(mamba): replace M_mu with selective scan architecture`
  - `fix(kalman): bind precision p_pred to residual variance`
  - `feat(ffl): add focal frequency loss with dj gating`
- Tag 规范：Commit 成功后，立刻打上局部标签，例如：
  `git tag -a v2.2-mamba-validated -m "Mamba integrated and overfit passed"`

## 5. 崩溃回退机制（时光机）

- 如果某个新模块加入后，导致训练崩溃（NaN / 无梯度 / 指标暴跌），且短时间无法修复，必须**立刻执行回退**：
  `git reset --hard <上一个成功的 Tag>`
- 铁律：**绝不在崩溃的代码上强行缝补**。回退到上一个健康节点，重新冷静思考。

---

Claude 必须以此为最高行为准则。在每次执行主人的代码修改指令前，必须先阅读并遵守此文件中的 Git 与测试规范。
