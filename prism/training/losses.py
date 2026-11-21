"""losses —— Q 监督（log 域，detach Δ/M）+ 下游去除损失 hook（R_θ 未接入，留接口）。

Q 监督（v9 定案 §3）：
    r_z      = X_z − (1−Δ_sg)X_{z-1} − Δ_sg·M_sg(X_{z-1}, Θ_z)   （干净 X，detach Δ/M）
    Q_target = Var_{窗口}[r]                                     （逐体素，5×5 in-plane × 3 z）
    L_Q      = mean_j (log Q_z − log Q_target)²                  （log 域）

Q 是逐体素场（非全局标量）。窗口物理尺寸 in-plane 2mm（5×5 @ 0.43mm voxel）、z ±1 slice；
重建网格固定 → voxel 尺寸恒定 → 窗口跨几何一致。λ_Q 起步 0.1（SSMConfig.lambda_q）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..models.r_theta import HU_SCALE
from ..models.m_network import LOGQ_MIN, LOGQ_MAX

# Q_target 局部方差窗口：in-plane 5×5（2mm @ 0.43mm voxel），z ±1（3 片）
Q_WINDOW = 5
Q_Z_WINDOW = 3
_EPS = 1e-12


def compute_clean_residual(X_z, X_prev, delta_sg, mu_forecast_sg):
    """干净残差 r_z = X_z − (1−Δ_sg)X_{z-1} − Δ_sg·M_sg(X_{z-1}, Θ_z)。

    X_z / X_prev: 干净组织切片 [B, 1, H, W]；delta_sg、mu_forecast_sg 已 detach。
    """
    mu_pred = (1.0 - delta_sg) * X_prev + delta_sg * mu_forecast_sg
    return X_z - mu_pred


def residual_to_q_target(residual, window: int = Q_WINDOW, z_window: int = Q_Z_WINDOW):
    """残差 → Q_target = Var_{窗口}，逐体素。

    residual: [B, 1, Z, H, W]。用 3D 平均池化求 E[r] 与 E[r²] 的局部差（Var）。
    """
    # C1 强递归（Z<3 序列进递归）：Q 窗口 kT=3 但实际片数不足。短序列退化为 in-plane 5×5
    # 方差（z 窗=1，不跨片），避免 avg_pool3d 在 T<3 时报错（input smaller than kernel），
    # 且奇数窗=1 保证输出 T 与输入一致（偶数窗 + pad//2 会撑大 Z 维）。
    Z = residual.shape[2]
    if Z < z_window:
        z_window = 1
    pad_z, pad_h, pad_w = z_window // 2, window // 2, window // 2
    mean = F.avg_pool3d(residual, kernel_size=(z_window, window, window),
                        stride=1, padding=(pad_z, pad_h, pad_w))
    mean_sq = F.avg_pool3d(residual ** 2, kernel_size=(z_window, window, window),
                           stride=1, padding=(pad_z, pad_h, pad_w))
    return (mean_sq - mean ** 2).clamp(min=_EPS)


def q_supervision_loss(log_q, q_target, mask=None):
    """Q 监督损失 L_Q = mean_j (log Q_z − log Q_target)²（log 域）。

    log_q: 网络 log 域输出 [B, 1, Z, H, W]；q_target: [B, 1, Z, H, W]。
    mask: 可选 [B, 1, Z, H, W]（非金属/组织掩膜 I：1=监督，0=金属遮盖，FIND-Net Eq.6）。

    防爆：log_q 先 clamp 到 [LOGQ_MIN, LOGQ_MAX]（与 forward 的 exp 前 clamp 一致）。
    g_Q 读 μ，μ 量级一旦涨（M_μ bias 不受 spectral_norm 约束），log_q 可冲到 ~3e4，
    原始平方损失 → Q~1e9、梯度~6e4，clip_grad 截断后反向振荡打乱结构学习。
    clamp 后损失≤(LOGQ_MAX−log_qt)²≈900、梯度在边界外为 0：μ 的瞬态过冲由 L_down
    （梯度受 HU_SCALE 归一化，O(0.1)）平滑拉回，Q 监督不添乱。
    """
    log_q = log_q.clamp(min=LOGQ_MIN, max=LOGQ_MAX)
    log_qt = torch.log(q_target + _EPS)
    se = (log_q - log_qt) ** 2
    if mask is not None:
        se = se * mask
        return se.sum() / (mask.sum() + _EPS)
    return se.mean()


def _ssim_loss(pred, target, mask=None):
    """结构相似度约束（高频结构项）：1 − mean(SSIM·mask)，越小越好。

    pred/target: [N, C, H, W]（已 ÷HU_SCALE 归一化）；mask: [N, C, H, W]（1=监督）。
    SSIM 用 11×11 高斯窗（σ=1.5），C1/C2 取标准 (0.01², 0.03²)（归一化后 L≈1）。
    惩罚「向均值坍塌」：纯 MSE 在盲区只逼均值，SSIM 项逼局部结构/纹理对齐。
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ws = 11
    c = torch.arange(ws, dtype=pred.dtype, device=pred.device) - (ws - 1) / 2
    g = torch.exp(-(c ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    win = (g[:, None] * g[None, :]).view(1, 1, ws, ws).to(pred.dtype)
    pad = ws // 2
    mu1 = F.conv2d(pred, win, padding=pad)
    mu2 = F.conv2d(target, win, padding=pad)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    s1 = F.conv2d(pred * pred, win, padding=pad) - mu1_sq
    s2 = F.conv2d(target * target, win, padding=pad) - mu2_sq
    s12 = F.conv2d(pred * target, win, padding=pad) - mu1 * mu2
    ssim = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / \
           ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2) + 1e-8)
    loss = (1.0 - ssim) * 0.5
    if mask is not None:
        return (loss * mask).sum() / (mask.sum() + _EPS)
    return loss.mean()


def _gradient_loss(pred, target, mask=None):
    """方向性梯度损失：分别算 x/y 方向 Sobel 梯度，返回 (loss_gx, loss_gy)。

    v1.5：显式解耦 x/y 方向（为后续按伪影放射方向加权预留接口）。
      loss_gx = L1(Sobel_x(pred), Sobel_x(target))
      loss_gy = L1(Sobel_y(pred), Sobel_y(target))
    注：v1.4 的实现在数学上已等价（gx+gy），v1.5 仅把两方向显式拆出。
    """
    sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    sy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                      dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    loss_gx = (F.conv2d(pred, sx, padding=1) - F.conv2d(target, sx, padding=1)).abs()
    loss_gy = (F.conv2d(pred, sy, padding=1) - F.conv2d(target, sy, padding=1)).abs()
    if mask is not None:
        loss_gx = (loss_gx * mask).sum() / (mask.sum() + _EPS)
        loss_gy = (loss_gy * mask).sum() / (mask.sum() + _EPS)
    else:
        loss_gx = loss_gx.mean()
        loss_gy = loss_gy.mean()
    return loss_gx, loss_gy


def downstream_reconstruction_parts(x_hat_z, X_clean, metal_mask=None, air_hu_thr: float = -500.0,
                                    zfreq_gain: float = 0.0, zfreq_scale: float = 50.0):
    """返回 L_down 的三个 masked 分量 (mse, ssim, grad)，供训练器打印诊断曲线。"""
    X_clean_p = X_clean.permute(0, 2, 1, 3, 4)          # [B,1,Z,H,W]
    resid = (x_hat_z - X_clean_p) / HU_SCALE
    if metal_mask is not None:
        mask = (1.0 - metal_mask).permute(0, 2, 1, 3, 4)     # [B,1,Z,H,W] 非金属
    else:
        mask = torch.ones_like(x_hat_z)
    mask = mask * (X_clean_p > air_hu_thr).float()      # 屏蔽空气，只留组织
    if zfreq_gain > 0.0:
        # exp008 频率加权（Sobolev 式，沿 z）：GT 的 z 一阶差分大的地方（高频结构）
        # 误差权重更高 —— 逼网络在 z 高频处也别抹平。权重上界 1+4·gain，防爆炸。
        gz = torch.zeros_like(X_clean_p)
        gz[:, :, 1:] = (X_clean_p[:, :, 1:] - X_clean_p[:, :, :-1]).abs()
        w = 1.0 + zfreq_gain * (gz / zfreq_scale).clamp(max=4.0)
        mask = mask * w
    mse = ((resid ** 2) * mask).sum() / (mask.sum() + _EPS)
    B, C, Z, H, W = x_hat_z.shape
    p = (x_hat_z / HU_SCALE).reshape(B * Z, C, H, W)
    t = (X_clean_p / HU_SCALE).reshape(B * Z, C, H, W)
    m = mask.reshape(B * Z, C, H, W)
    ssim = _ssim_loss(p, t, m)
    grad_x, grad_y = _gradient_loss(p, t, m)
    grad = grad_x + grad_y       # 总 Loss = MSE + λ_ssim·SSIM + λ_grad·(loss_gx + loss_gy)
    return mse, ssim, grad


def downstream_reconstruction_loss(x_hat_z, X_clean, metal_mask=None, air_hu_thr: float = -500.0,
                                   lambda_ssim: float = 0.25, lambda_grad: float = 0.1):
    """L_down = masked_MSE + λ_ssim·(1−SSIM) + λ_grad·SobelGrad —— 去伪影 + 高频结构 + 边缘硬化。

    递归循环定案：x̂_z = FIND-Net(CT_ma=脏图, LIct=μ_pred, Mask) 是唯一最终输出，非金属区 I 监督。
    残差按 HU_SCALE=100 归一化。梯度经 x̂_z 回流：x̂_z 读 μ_pred → M_μ；观测经卡尔曼混合进 carry。
    v1.1：屏蔽空气；v1.3：SSIM 高频结构；v1.4：λ_ssim 0.1→0.25，新增 Sobel 梯度损失
    （λ_grad=0.1）逼网络在骨/气硬边界恢复锐利边缘，杜绝鼻窦/颅底/乳突气房的过度平滑。

    x_hat_z: [B,1,Z,H,W]（FIND-Net 输出，HU）；X_clean: [B,Z,1,H,W]；metal_mask: [B,Z,1,H,W]（1=金属）。
    """
    mse, ssim, grad = downstream_reconstruction_parts(x_hat_z, X_clean, metal_mask, air_hu_thr)
    return mse + lambda_ssim * ssim + lambda_grad * grad


def mu_fidelity_loss(mu, X_clean, metal_mask=None):
    """L_mu = mean_{非金属} (μ − X_clean)² —— 组织先验 μ 的直接保真监督。

    诊断根因（eval_mu_quality.py）：μ 从无「μ ≈ X_clean」的直接项，只有 L_down 经
    x̂_final = μ + corr 的 ∂/∂μ=1 干净路径间接拉 μ。这存在退化：x̂_final = X_clean 可被
    任意 (μ, corr) 满足（μ+corr=X_clean），L_corr 只能弱破退化，R_θ 便宜地把信号吸进
    corr、μ 退化成平坦先验。实测：部分阴影区 μ_RMSE(328) 比原始观测 x̂(302) 还差，
    完全阴影区 μ 与 X_clean 零相关（corr≈−0.06）——递归没在传组织，μ 被架空。

    L_mu 直接逼 μ≈X_clean（非金属区，含阴影区），消除退化：在完全阴影区 M_μ 读 d_j≈0
    是盲的，唯一能凑出 X_clean 的办法就是让门控 Δ→0、把 μ_{z-1} 的组织 carry 进来
    ——从而逼递归真正工作。这正是「干净先验 + 递归填阴影」理论链条缺失的那一环。

    mu: [B,1,Z,H,W]（组织先验，HU）；X_clean: [B,Z,1,H,W]（干净组织，HU）；
    metal_mask: [B,Z,1,H,W]（1=金属）。残差按 HU_SCALE 归一化（与 L_down 同尺度）。
    """
    resid = (mu - X_clean.permute(0, 2, 1, 3, 4)) / HU_SCALE
    se = resid ** 2
    if metal_mask is not None:
        mask = (1.0 - metal_mask).permute(0, 2, 1, 3, 4)     # [B,1,Z,H,W]
        se = se * mask
        return se.sum() / (mask.sum() + _EPS)
    return se.mean()


def physics_consistency_loss(mu, d_j, d_ref, metal_hu_thr: float = 2500.0,
                             d_thr_frac: float = 0.5):
    """L_pc —— 物理一致性惩罚：惩罚「模型输出金属 HU，但观测充分度 d_j 却很高」的体素。

    Physics consistency（第一性原理）：真金属 ⟺ 光子饥饿 ⟺ d_j 骤降 + 大范围阴影。
    牙齿/骨 HU 虽高但没有光子饥饿（d_j 不塌），所以「金属 + 高 d_j」= 物理矛盾。
    该惩罚把这条物理常识变成可微信号，逼模型在内部打假。

    **关键修正（为什么不能用冻结输入 x_M0）**：x_M0 与 d_j 都是冻结输入（detach，
    M 网络只读不更新），直接用它们算 L_pc 时 dL_pc/dθ ≡ 0 —— 那是死项。因此惩罚必须
    绑定在**模型输出** μ 上：用 μ 派生「模型金属信念」x̂_M0 = relu(μ − metal_hu_thr)，
    结构与你给的分式一致，只是把 x_M0 换成可微的 x̂_M0。

        x̂_M0      = relu(μ − metal_hu_thr) / metal_hu_thr     （模型输出的金属 HU）
        d_photon  = relu(d_j / d_ref − d_thr_frac)             （光子充足度）
        L_pc      = mean_j (x̂_M0,j · d_photon,j)²

    真金属区 d_j 塌 → d_photon=0 → 不罚（正确放行金属）；假金属区（牙齿/骨）d_j 高 →
    d_photon>0 → 若模型还输出金属 HU 就狠狠惩罚。

    Parameters
    ----------
    mu: [B,1,H,W] 或 [B,1,Z,H,W] 模型干净估计（HU）。
    d_j: 同形，原始数据充分度（未归一化，AGIR d_j = AᵀW(A·1)）。
    d_ref: 冻结标量（median_j d_j，C5）。d_thr = d_thr_frac · d_ref（默认 0.5·d_ref）。
    metal_hu_thr: 判定「金属 HU」阈值（真牙/致密骨 <2500，Pt-W 远高于此）。
    d_thr_frac: d_thr 相对 d_ref 的比例（默认 0.5）。

    Returns
    -------
    标量 tensor（requires_grad=True，经 μ 回流）。
    """
    metal_belief = torch.relu(mu - metal_hu_thr) / metal_hu_thr
    photon_sufficient = torch.relu(d_j / d_ref - d_thr_frac)
    penalty = (metal_belief * photon_sufficient) ** 2
    return penalty.mean()
