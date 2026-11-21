"""trainer —— 三轴训练循环（multi-dose × multi-geometry × multi-patient，v9 §4）。

递归循环定案（prism-interface-loss-final）：FIND-Net 是主、唯一输出者 x̂_z；SSM 是宾，
先预测 μ_pred（carry 先验）喂给 FIND-Net 输入，FIND-Net 输出 x̂_z 又作为 SSM 观测喂回
卡尔曼混合 μ_post = K·μ_pred + (1−K)·x̂_z，carry 到 z+1。观测 = FIND-Net 输出 x̂_z，
**不是** Baseline 脏图（脏图是 FIND-Net 自己的 CT_ma 输入）。

loss 定案（三主）：L_down（x̂_z ≈ X_clean，去伪影）+ L_mu（μ ≈ X_clean，干净引导）+
L_Q（过程噪声标定）；L_corr 已删（它是旧「x̂_final=μ+corr」错误接口的产物）；L_pc 待定。

三轴都是默认（非兜底）：剂量 {0.25,0.5,1,2,4}× 对数间隔、几何 SID/SOD ±20%、多患者。
物理量（W/d_j/g/tau_b/w_b）全部 detach——从数据进入，M 网络只读不更新。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..recursion.algorithm1 import forward_backward
from ..kalman.state import KalmanState, total_precision
from ..models.find_net import FINDNet, FINDNetArgs
from ..utils.normalize import compute_d_ref
from ..utils.config import SSMConfig
from . import losses


class SSMTrainer:
    def __init__(self, m_net, config=None, optimizer=None, device="cpu",
                 find_net=None, r_theta=None):
        self.m_net = m_net
        self.config = config if config is not None else SSMConfig(nb=m_net.nb)
        self.device = device
        self.optimizer = optimizer
        # 递归循环定案：下游 = 真实 FIND-Net（替代旧 R_θ 3 层 conv 替身）。
        # r_theta 参数仅向后兼容旧诊断脚本（存引用，train_step 不再用它）。
        self.find_net = (find_net if find_net is not None
                         else FINDNet(FINDNetArgs()).to(device))
        self.r_theta = r_theta

    def parameters(self):
        """M 网络 + FIND-Net 的全部可训练参数（优化器 / 梯度裁剪共用）。"""
        return list(self.m_net.parameters()) + list(self.find_net.parameters())

    @staticmethod
    def freeze_d_ref(config, d_j_maps) -> float:
        """从训练集全部 d_j 场冻结 d_ref = median_j d_j（C5），写回 config.d_ref。"""
        config.d_ref = compute_d_ref(d_j_maps)
        return config.d_ref

    @staticmethod
    def _stack(xs):
        """[Z] 个 [B, 1, H, W] → [B, 1, Z, H, W]。"""
        return torch.stack(xs, dim=2)

    def _q_target(self, thetas, X_clean, fwd_states, fwd_delta):
        """Q_target = Var_{窗口}[X_z − (1−Δ_sg)X_{z-1} − Δ_sg·M_sg(X_{z-1}, Θ_z)]。

        X_clean: [B, Z, 1, H, W] 干净组织。首片（z=0 冷启动）无残差 → 置 0（靠 mask 遮盖）。
        返回 [B, 1, Z, H, W]。
        """
        Z = len(thetas)
        res = [None] * Z
        for z in range(Z):
            if z == 0:
                res[z] = torch.zeros_like(X_clean[:, 0, ...])
                continue
            X_z = X_clean[:, z, ...]
            X_prev = X_clean[:, z - 1, ...]
            dmu = (torch.zeros_like(X_prev) if z == 1
                   else X_clean[:, z - 1, ...] - X_clean[:, z - 2, ...])
            p_prior = torch.full_like(fwd_states[z - 1].p_post, self.config.fixed_prior_p)
            S_clean = KalmanState(mu=X_prev, p_post=p_prior, delta_mu=dmu)
            mu_forecast_sg = self.m_net.forecast(S_clean, thetas[z], self.config.d_ref).detach()
            delta_sg = fwd_delta[z].detach()
            res[z] = losses.compute_clean_residual(X_z, X_prev, delta_sg, mu_forecast_sg)
        return losses.residual_to_q_target(torch.stack(res, dim=2))

    def _trust_signal(self, mu_fused, p_post_fused, d_j):
        """R_θ 第 3 通道的信任信号（C 消融 H2：门控 vs 卡尔曼 vs 固定）。

        递归循环定案后 FIND-Net 不再吃显式 trust 通道（信任隐含在 μ_pred/K 里），本方法
        仅保留给旧 C 消融诊断脚本（eval_knob_sweep 等）复用。
        """
        mode = self.config.trust_mode
        if mode == "kalman":
            return total_precision(p_post_fused)
        if mode == "gated":
            _, _, Z, _, _ = mu_fused.shape
            trust_z = [self.m_net.compute_gated_trust(
                mu_fused[:, :, z, :, :], d_j[:, :, z, :, :], self.config.d_ref
            ) for z in range(Z)]
            return torch.stack(trust_z, dim=2)
        if mode == "fixed":
            return self.m_net.fixed_trust.expand_as(mu_fused)
        raise ValueError(f"未知 trust_mode: {mode!r}")

    def _reconstruct(self, dirty_z, mu, mask):
        """FIND-Net 前向的「整网」梯度检查点（观测与最终输出共用）。

        两层检查点配合（缺一在 512×512 联合训练都会 OOM）：
          - 内层（find_net.py 逐 ProxNet）：把单次 FIND-Net 前向图从 ~11GB 压到 ~1.84GB；
          - 外层（本方法整网）：让递归里每片一次 FIND-Net 前向在 forward 阶段不持图
            （只存输入+输出 ~3MB），反向时按片重算（~2GB 瞬时，内层再压）。
        否则递归 3Z 次 FIND-Net 全展开同图 → max-z=16 需 ~88GB。MNetwork.predict 图小
        （~170MB/次，2Z 次 ≈ 5GB@Z=16）不检查点，保持实现简单。
        """
        return torch.utils.checkpoint.checkpoint(
            self.find_net.reconstruct, dirty_z, mu, mask, use_reentrant=False)

    def _lict(self, mu, li_z):
        """FIND-Net 的 LIct 输入（A2 去先验）：默认 μ（SSM 先验）；"li" 时用经典 LI-MAR。"""
        if self.config.findnet_lict_source == "li" and li_z is not None:
            return li_z
        return mu

    def _observer(self, dirty, masks, li=None):
        """FIND-Net 观测闭包：μ_pred → x̂_z = FIND-Net(CT_ma=脏图, LIct=μ_pred|LI, Mask)。"""
        def obs(mu_pred, z):
            mask = masks[z] if masks is not None else None
            lict = self._lict(mu_pred, li[z] if li is not None else None)
            return self._reconstruct(dirty[z], lict, mask)
        return obs

    def _backfill(self, pred, baseline, metal_mask):
        """v1.3 金属回填：Final = (1−Mask_dilate_r)·Pred + Mask_dilate_r·Baseline。

        metal_mask: [B,1,H,W]（1=金属）。二值化后形态学膨胀 r px 盖住金属边缘晕影，
        金属核内部原样保留 Baseline 的物理高光（HU>3000），网络绝不在金属区生成组织。
        r≤0 或 metal_mask=None 时不回填（消融 B2/旧行为）。
        """
        r = self.config.backfill_r
        if metal_mask is None or r <= 0:
            return pred
        binm = (metal_mask > 0.5).float()
        k = 2 * r + 1
        dil = F.max_pool2d(binm, kernel_size=k, stride=1, padding=r)
        return (1.0 - dil) * pred + dil * baseline

    def _final_output(self, res, dirty, masks, li=None):
        """FIND-Net 唯一输出 x̂_z（L_down 监督对象），[B, 1, Z, H, W]，含 v1.3 金属回填。

        双向：先融合 SSM 后验 μ_post（精度加权），再 FIND-Net(脏图, 融合 μ|LI, Mask) ——
        「融合先验 → FIND-Net → 最终输出」，后向趟的贡献经融合 μ 进最终输出。
        单向/per_slice：直接用循环内 FIND-Net 输出 res.fwd_x_hats（已是 FIND-Net(μ_pred|LI)）。
        两种模式最终输出都做金属回填，保证评估与训练逻辑对齐。
        """
        if self.config.recursion == "bidirectional":
            mu_fused = [s.mu for s in res.fused]               # [Z] 个 [B,1,H,W]
            xs = []
            for z in range(len(mu_fused)):
                mask = masks[z] if masks is not None else None
                lict = self._lict(mu_fused[z], li[z] if li is not None else None)
                pred = self._reconstruct(dirty[z], lict, mask)
                xs.append(self._backfill(pred, dirty[z], mask))
            return self._stack(xs)
        # 非双向：循环内 FIND-Net 输出，同样回填
        xs = []
        for z in range(len(res.fwd_x_hats)):
            mask = masks[z] if masks is not None else None
            xs.append(self._backfill(res.fwd_x_hats[z], dirty[z], mask))
        return self._stack(xs)                                 # [B,1,Z,H,W]

    def train_step(self, thetas, X_clean, metal_mask, x_hat, clip_grad: float = 0.0, li=None):
        """一个体的训练步（递归循环：FIND-Net 在循环内，观测 = FIND-Net 输出）。

        thetas: [Z] dict（物理量，已 detach）；X_clean / metal_mask / x_hat: [B, Z, 1, H, W]
        （x_hat = Baseline 脏图，是 FIND-Net 的 CT_ma 输入，不是 SSM 观测）。
        li: [B, Z, 1, H, W] 经典 LI-MAR（A2 去先验时 FIND-Net 的 LIct 输入；None=不用）。
        返回 (total, q_loss, pc_loss, down_loss, mu_loss) 五个标量 tensor。
        """
        Z = len(thetas)
        # B1 硬切掩膜：金属掩膜二值化 0/1（默认连续 vf），作用于 FIND-Net Mask 输入与损失掩膜。
        if self.config.binarize_mask and metal_mask is not None:
            metal_mask = (metal_mask > self.config.mask_bin_thr).float()
        x_hat = x_hat.permute(0, 2, 1, 3, 4)                       # [B,1,Z,H,W]
        metal_mask_4d = (None if metal_mask is None
                         else metal_mask.permute(0, 2, 1, 3, 4))   # [B,1,Z,H,W]
        dirty = [x_hat[:, :, z] for z in range(Z)]                 # [Z] 个 [B,1,H,W]
        masks = ([metal_mask_4d[:, :, z] for z in range(Z)]
                 if metal_mask_4d is not None else None)
        li_4d = li.permute(0, 2, 1, 3, 4) if li is not None else None
        li_list = ([li_4d[:, :, z] for z in range(Z)] if li_4d is not None else None)
        observer = self._observer(dirty, masks, li_list)

        # 冷启动暖启动（Stage 2 短序列强拉收敛）：S0.μ 用首片观测而非全零。
        if self.config.cold_start_init == "obs":
            cold_start_mus = [d.detach() for d in dirty]           # Baseline 脏图（组织观测）
        elif self.config.cold_start_init == "xm0":
            cold_start_mus = [t["x_M0"].detach() for t in thetas]  # 金属先验（非组织，慎用）
        else:
            cold_start_mus = None

        # 1. 全程递归展开（FIND-Net 在循环内产观测）
        res = forward_backward(self.m_net, thetas, self.config.d_ref,
                               recursion=self.config.recursion, observer=observer,
                               cold_start_mus=cold_start_mus,
                               gain_mode=self.config.gain_mode)

        # 非金属/组织掩膜 I = 1−metal_mask（1=监督，FIND-Net Eq.6）
        if metal_mask is not None:
            mask = (1.0 - metal_mask).permute(0, 2, 1, 3, 4)       # [B,1,Z,H,W]
        else:
            mask = None

        # B1：金属内算 Loss（只作用于重建损失 L_down/L_mu；Q 监督仍弃疗金属——Q_target
        # 是组织方差，金属区无真组织信号，监督无意义）。
        loss_metal_mask = None if self.config.loss_include_metal else metal_mask

        # 2. Q 监督（前向趟）
        q_target = self._q_target(thetas, X_clean, res.fwd, res.fwd_delta)
        log_q = self._stack(res.fwd_logq)                          # [B,1,Z,H,W]
        q_loss = losses.q_supervision_loss(log_q, q_target, mask)

        # 3. L_down：FIND-Net 唯一输出 x̂_z ≈ X_clean（去伪影，主）
        x_hat_z = self._final_output(res, dirty, masks, li_list)   # [B,1,Z,H,W]
        self.last_down_parts = losses.downstream_reconstruction_parts(
            x_hat_z, X_clean, loss_metal_mask, zfreq_gain=self.config.zfreq_gain)
        _mse, _ssim, _grad = self.last_down_parts
        down_loss = (_mse + self.config.lambda_ssim * _ssim + self.config.lambda_grad * _grad)

        # 4. L_pc（待定，保留现状）：μ 的金属 HU 物理一致性
        mu_fused = self._stack([s.mu for s in res.fused])          # [B,1,Z,H,W]
        d_j = self._stack([t["d_j"] for t in thetas])              # [B,1,Z,H,W]
        pc_loss = losses.physics_consistency_loss(
            mu_fused, d_j, self.config.d_ref,
            metal_hu_thr=self.config.pc_metal_hu_thr,
            d_thr_frac=self.config.pc_d_thr_frac)

        # 5. L_mu：组织先验 μ ≈ X_clean（干净引导，监督本片；盲区供 FIND-Net 参考）
        mu_loss = losses.mu_fidelity_loss(mu_fused, X_clean, loss_metal_mask)

        total = (self.config.lambda_q * q_loss
                 + self.config.lambda_pc * pc_loss
                 + self.config.lambda_down * down_loss
                 + self.config.lambda_mu * mu_loss)
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            total.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)
            self.optimizer.step()
        return total, q_loss, pc_loss, down_loss, mu_loss
