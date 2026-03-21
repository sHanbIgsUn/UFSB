import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class UnbalancedOptimalTransport(nn.Module):
    def __init__(self, eps: float = 1.0, tau_start: float = 1.0, tau_end: float = 1.0, total_epochs: int = 100,
                 max_iter: int = 100):
        """
        Args:
            eps (float): 熵正则化系数 (Epsilon)
            tau_start (float): 初始 tau 
            tau_end (float): 最终 tau 
            total_epochs (int): 总训练轮数
            max_iter (int): Sinkhorn 迭代次数
        """
        super(UnbalancedOptimalTransport, self).__init__()
        self.eps = eps
        self.max_iter = max_iter

        # --- 自适应 Tau 参数 ---
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.total_epochs = total_epochs

        # 初始化当前 tau
        self.current_tau = tau_start

    def update_tau(self, epoch, strategy='cosine'):
 
        epoch = min(epoch, self.total_epochs)
        progress = epoch / self.total_epochs

        if strategy == 'linear':
            # 线性衰减
            self.current_tau = self.tau_start - (self.tau_start - self.tau_end) * progress

        elif strategy == 'exp':
            # 指数衰减
            if self.tau_end <= 0: self.tau_end = 1e-4  # 
            decay_rate = (self.tau_end / self.tau_start) ** (1 / self.total_epochs)
            self.current_tau = self.tau_start * (decay_rate ** epoch)

        elif strategy == 'cosine':
            # 余弦退火
            self.current_tau = self.tau_end + 0.5 * (self.tau_start - self.tau_end) * \
                               (1 + math.cos(math.pi * progress))

        return self.current_tau

    def solve_unbalanced_sinkhorn(self, z1, z2):
        """
        计算两个分布 z1 和 z2 之间的 Unbalanced OT Cost
        """
        # 1. 归一化
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)

        # 2. 计算 Cost Matrix (Cost 范围 [0, 4])
        cost = torch.cdist(z1, z2, p=2) ** 2

        # 3. 初始化变量
        b1, b2 = z1.shape[0], z2.shape[0]

        # 假设均匀分布
        mu = torch.empty(b1, dtype=z1.dtype, device=z1.device).fill_(1.0 / b1)
        nu = torch.empty(b2, dtype=z2.dtype, device=z2.device).fill_(1.0 / b2)

        u = torch.ones_like(mu)
        v = torch.ones_like(nu)

        # 4. 预计算 Kernel K
        K = torch.exp(-cost / self.eps)

        # 5. Unbalanced Sinkhorn 迭代
        # 使用动态的 self.current_tau 计算缩放因子 fi
        # tau 越大 -> fi 接近 1 (Balanced)
        # tau 越小 -> fi 接近 0 (Unbalanced, Mass Destruction)
        fi = self.current_tau / (self.current_tau + self.eps)

        for _ in range(self.max_iter):
            # 迭代公式：增加了 ** fi 的指数缩放
            # 加 1e-8 防止除零
            u = (mu / (torch.matmul(K, v) + 1e-8)) ** fi
            v = (nu / (torch.matmul(K.T, u) + 1e-8)) ** fi

        # 6. 计算 Transport Plan
        # P = diag(u) * K * diag(v)
        transport_plan = torch.diag(u) @ K @ torch.diag(v)

        # 7. 计算 Transport Cost
        loss = torch.sum(transport_plan * cost)

        return loss

    def forward(self, z_s, z_t):
        # 兼容 list 输入
        if isinstance(z_s, list): z_s = z_s[0]
        if isinstance(z_t, list): z_t = z_t[0]

        # 统一归一化
        z_s = F.normalize(z_s, p=2, dim=1)
        z_t = F.normalize(z_t, p=2, dim=1)

        # --- Sinkhorn Divergence 计算 ---
        # S(a,b) = OT(a,b) - 1/2*OT(a,a) - 1/2*OT(b,b)

    
        cost_st = self.solve_unbalanced_sinkhorn(z_s, z_t)
        cost_ss = self.solve_unbalanced_sinkhorn(z_s, z_s)
        cost_tt = self.solve_unbalanced_sinkhorn(z_t, z_t)

        loss = cost_st - 0.5 * cost_ss - 0.5 * cost_tt

        return loss
