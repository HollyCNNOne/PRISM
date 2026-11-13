# PRISM 实验与版本记录（EXPERIMENT_LOG）

> 规范：每次 Commit / 实验按下方 10 项模板追加一条；「跑成功的代码 = 一条记录 + 一个新 Tag」。
> 最新在前（倒序）。

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
