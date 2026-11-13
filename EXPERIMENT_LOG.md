# PRISM 实验与版本记录（EXPERIMENT_LOG）

> 规范：每次 Commit / 实验按下方 10 项模板追加一条；「跑成功的代码 = 一条记录 + 一个新 Tag」。
> 最新在前（倒序）。

---

## v1.1-scale-fix

- [版本号] v1.1-scale-fix
- [时间] 2026-11-13 22:23
- [实验前提与条件] 代码考古（vs 官方 FIND-Net-main）确认三处契约：① 输入 (x+1500)/5000 → clip[0,1] → **×255** → [0,255]；② 输出 X/255 → ×5000 −1500；③ Mask = non_Mask（**1=非金属**，0=金属）。我 v1.1 的实现丢掉了 ×255、把 Mask 极性做成 1=金属；且此前的「方案 A 归一化」已被证伪——它只把死掉的 0 映射成 −1500，没解决 Dead ReLU。
- [本次目的] 控制变量验证：只改「尺度放大 ×255 + Mask 极性还原」这两项，看 Dead ReLU / Mode Collapse 是否解除、组织是否长出来（GroupNorm 与 `F.relu(X+layer(X))` 一律不动，留待 v1.1+）。
- [采用方法] ① find_net.forward 输入 `clip((x+1500)/5000,0,1)*255`；② reconstruct 输出 `X/255*5000−1500`；③ 传入 FINDNet 的 Mask 改 1=非金属（caller 传 `1−metal_mask`）；clamp 域同步 [0,1]→[0,255]。10 步 overfit（large/giant 两片，同 seed）。
- [实验结果] 10 步：loss 43.2→25.4 单调降；rawXmax=215.8（[0,255] 域内，**无爆炸**）；HU min/max=−1500/2672（物理量程内）；组织区 RMSE=547/459（baseline 765/649，**跌破 218/190**）；corr=+0.471/+0.586（baseline +0.347/+0.461，**转正且持续上升**）；负像素 215889（能画空气）。
- [实验结果·200 步验收] loss 43.2→2.0 单调降（step1→50→100→200：43.2→8.0→2.9→2.0）；组织区 RMSE=131/142（baseline 765/649，**5.8×/4.6× 提升**）；corr=+0.938/+0.945（baseline +0.347/+0.461）；rawXmax=175.9、HU max=1949（无爆炸）；负像素 184031/162372（能画空气）。Prediction 星芒显著削弱、向 Target 靠拢，Error 热图明显变暗。
- [成功点] ✅ **成功定案（Resolved）**：**通过输入 ×255 解决残差死锁，Loss 正常收敛，模型具备去伪影能力**。尺度放大 + Mask 极性还原解除 Dead ReLU / 塌缩：组织区 RMSE 跌破 baseline 且 corr 转正逼近 0.94，无数值爆炸、空气可表示。主人的「GroupNorm 破坏作者权重分布」判断得到控制变量证实。
- [失败点/异常] 无（200 步全收敛，无 NaN、无爆炸）。
- [问题根因分析] Dead ReLU 直接诱因 = 「输入动态范围 [0,1] 过小 + GroupNorm 零均值」：×255 把残差跳跃里的 X 放大 255 倍，`X+layer(X)` 不易被负残差拉穿 ReLU；Mask 极性还原让伪影从非金属区提取（对齐官方算法语义）。
- [下一步状态] ✅ 成功定案（Resolved）。v1.1+（删 Xnet/Mnet 的 `F.relu(X+layer(X))`）暂缓，作为后续可选优化（届时 ×255/÷255 翻译服也可弃用）。

---

## v1.1-fix-hu-contract

- [版本号] v1.1
- [时间] 2026-11-13 20:32
- [实验前提与条件] 基于 v1.0 发现的两大致命冲突：① FIND-Net Xnet `F.relu` 钳输出≥0 → 网络缺乏负数输出能力（空气 −1000 HU 不可表示）；② 物理 HU 真实量纲与黑盒归一化假设不和谐（原版 FIND-Net 假设 [0,1] 输入，PRISM 直喂原始 HU）。
- [本次目的] 解决物理真实性与 FIND-Net 黑盒架构的不兼容：验证「标准化翻译服（归一化+反归一化）+ 背景屏蔽」能否解决 Mode Collapse，并唤醒模型对高频组织的学习能力。
- [采用方法] ① DataLoader/网络接口处强制 `(HU+1500)/5000` 映射，网络输出后反映射回 HU；② B_down Loss 中彻底屏蔽空气（X_clean < −500）与金属核心（1−metal_mask）。
- [实验结果] 10 步 overfit：loss 115865→298.9 单调降、无 NaN；负像素数 246845（94%，ReLU 钳死解除、能输出空气）。但组织区没长出来：pred 组织区 RMSE=1702 > baseline 765，corr=−0.02（负相关），pred 组织区 mean=−1491（组织也被画成空气）。
- [成功点] 翻译服（归一化+反归一化）生效：网络恢复负数输出能力（v1.0 负像素数=0）。
- [失败点/异常] 数值爆炸（Numerical Explosion）：pred max HU=277533、min=−1500，超出物理量程。
- [问题根因分析] 三刀解剖：① FIND-Net Xnet `F.relu(X+layer(X))` 输出下界 0、上界**无界**，探针实测 raw X max=+141（远超 [0,1] 契约）；② 反归一化 ×5000 放大：+141→705811 HU；③ v1.1 把空气（target≤−500，占 65%）mask 掉 → 空气区无监督 → FFC 边界角点 + 无界 ReLU 自由增长（爆炸体素 7/7 全在空气区，max 在 (0,0) 角点 tgt=−991）。梯度裁剪（grad_norm 772 万→1.0）挡不住，因爆炸来自「无上界输出映射」而非「梯度发散」（loss 单调降、无 NaN 可证）。
- [下一步状态] 修：forward 输出 `clamp(0,1)` 锁翻译服契约（反归一化锁定 [-1500,3500]）+ 梯度裁剪（trainer 已有，overfit 测试补上）。修后重跑 overfit 验证组织是否长出来。

---

## v1.0-baseline-collapse

- [版本号] v1.0
- [时间] 2026-11-13
- [实验前提与条件]
  - 数据集：training_data_astra（解析 ASTRA/skimage 平行束，1386 片；holdout=patients_head2 零泄漏；d_ref=4.100e6 冻结）
  - 架构：FIND-Net（10-stage FFC，Xnet/Mnet 近端网络）+ SSM M-network 递归高斯（双向 recursion）
  - SSM 开关：开（bidirectional 递归，FIND-Net 在循环内当观测，Kalman 混合）
- [本次目的] 完整管线首跑：Stage1 FIND-Net 预训练 → Stage2 八臂消融矩阵，先看 00_baseline 第一臂。
- [采用方法] 现状代码原样跑（无改动）：Stage1 200 epoch（早停）→ Stage2 消融；评估用 eval_mar.py 三分区（d_j/d_ref）。
- [实验结果]
  - 过渡区 RMSE=239.5 / Bias=−6.9 / PSNR=24.9
  - 充分区 RMSE=995.96（n=4.5M），部分阴影 723.9，完全阴影 1002.8，全局 804.2，peak 4228.7
  - B_down loss 卡 52.7→53.0 不降；Stage1 val_loss 平台 57.96→57.19（早停 epoch 11/200）
  - 预测图全灰（pred min=0.00 / mean≈0.09）；Stage1 2D 预测 pred min=0.00、负像素数=0
- [成功点] 过渡区 239.5 / Bias −6.9：SSM 卡尔曼核心机制（递归先验填过渡区）成立。
- [失败点/异常] Mode collapse：预测全灰，充分区 995、全局 804；FIND-Net 把充分区 baseline（RMSE 7.35）毁成 997。
- [问题根因分析]
  - 主因：FIND-Net Xnet `F.relu(X+layer(X))` 把重建输出钳到 ≥0，空气(−1000 HU)永远画不出；充分区(r>9)恰是空气主导（target mean −997）→ 误差≈1000 = 995。原始 HU 直入直出、无归一化，ReLU 在原版是为 [0,1] 归一化 CT 设计的。
  - 放大器：损失掩膜只排除金属(mask=1−metal)，空气仍在梯度里，1000 HU 误差把损失地板焊死。
  - 次因：μ 先验同样塌成近平坦（充分区 μ_corr=−0.006、μ_bias=+952、μ≈−45），但 μ clamp=[−1200,3200] 可负、非 ReLU 问题；FIND-Net 2D 阶段已坏（pred min=0.00），证明主因在 FIND-Net 不在 SSM。
- [下一步状态] 定手术方案后重训：A=HU→[0,1] 归一化/反归一化（推荐，匹配原版）；B=去掉 Xnet 重建路径 ReLU。未修复前禁止启动第 2–8 臂。
