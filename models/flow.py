import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class PlanarFlow(nn.Module):
    '''
    Planar Flow (Page 5, Section 4.1)
    '''
    def __init__(self, 
                 latent_dim: int,
                 use_params: bool = False,
                 invertibility_condition: bool = True):
        super().__init__()
        
        self.u, self.w, self.b = None, None, None
        self.u = nn.Parameter(torch.randn(latent_dim))
        self.w = nn.Parameter(torch.randn(latent_dim))
        self.b = nn.Parameter(torch.randn(1))
        
        # Weight initialization
        lim_w = np.sqrt(2.0 / latent_dim)
        nn.init.uniform_(self.w, -lim_w, lim_w)
        lim_u = np.sqrt(2.0)
        nn.init.uniform_(self.u, -lim_u, lim_u)
        nn.init.zeros_(self.b)

        self.h = nn.Tanh()

        self.invertibility_condition = invertibility_condition
        self.use_params = use_params

    def forward(self, z, u=None, w=None, b=None):
        # batch 단위로 내적하려면, 어쩔 수 없이 z가 앞에 와야 함. z @ self.w (수식 그대로 따라가고 싶은데 아쉽.)
        u, w, b = self.u, self.w, self.b

        f_z = self.h((z @ w) + b)

        if self.invertibility_condition == True:
            u_hat = self.get_uhat(w, u)
            f_z = z + u_hat * f_z.unsqueeze(-1)
        else:
            f_z = z + u * f_z.unsqueeze(-1)

        return f_z
    
    def diff_tanh(self, x):
        return 1 - torch.tanh(x) ** 2
    
    def log_abs_det_jacobian(self, z, u=None, w=None, b=None):
        u, w, b = self.u, self.w, self.b
        psi_z = self.diff_tanh((z @ w) + b).unsqueeze(-1) * w # (B, D)

        if self.invertibility_condition == True:
            u_hat = self.get_uhat(w, u)
            log_abs_det = torch.log(torch.abs(1 + (psi_z @ u_hat)))
        else:
            log_abs_det = torch.log(torch.abs(1 + (psi_z @ u)))

        return log_abs_det

    def get_uhat(self, w, u):
        # 가역성 조건 만족시키기 (Appendix A)
        w_u = w @ u
        # print(w.shape, u.shape, w_u.shape); exit()
        
        # use softplus for numerical stability
        # softplus(x) = log(1 + exp(x))
        m_wu = -1.0 + F.softplus(w_u)
        u_hat = u + ((m_wu - w_u) * w) / (w @ w)
        return u_hat

if __name__ == '__main__':
    flow = PlanarFlow(latent_dim=40)
    z = torch.randn(16, 40) 
    z_transformed = flow(z)

    flow.log_abs_det_jacobian(z)