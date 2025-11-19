import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class PlanarFlow(nn.Module):
    '''
    Planar Flow (Page 5, Section 4.1)
    '''
    def __init__(self, 
                 invertibility_condition: bool = True):
        super().__init__()

        self.h = nn.Tanh()

        self.invertibility_condition = invertibility_condition

    def forward(self, z, u=None, w=None, b=None):
        # batch 단위로 내적하려면, 어쩔 수 없이 z가 앞에 와야 함. z @ self.w (수식 그대로 따라가고 싶은데 아쉽.)
        '''
        z: (B, D)
        u: (B, D, 1)
        w: (B, 1, D)
        b: (B, 1, 1)
        '''

        z = z.unsqueeze(2) # (B, D) -> (B, D, 1)
        f_z = self.h(torch.bmm(w, z) + b) # (B, 1, 1)

        if self.invertibility_condition == True:
            u_hat = self.get_uhat(w, u) # (B, D, 1)

            f_z = z + u_hat * f_z # (B, D) + (B, D)
            f_z = f_z.squeeze(2)

        else:
            f_z = z + u * f_z
            f_z = f_z.squeeze(2)

        return f_z
    
    def diff_tanh(self, x):
        return 1 - torch.tanh(x) ** 2
    
    def log_abs_det_jacobian(self, z, u=None, w=None, b=None):
        psi_z = self.diff_tanh((z @ w) + b).unsqueeze(-1) * w # (B, D)

        if self.invertibility_condition == True:
            u_hat = self.get_uhat(w, u)
            log_abs_det = torch.log(torch.abs(1 + (psi_z @ u_hat)))
        else:
            log_abs_det = torch.log(torch.abs(1 + (psi_z @ u)))

        return log_abs_det

    def get_uhat(self, w, u):
        # 가역성 조건 만족시키기 (Appendix A)
        w_u = torch.bmm(w, u) # (B, 1, 1)
        
        # use softplus for numerical stability
        # softplus(x) = log(1 + exp(x))
        m_wu = -1.0 + F.softplus(w_u) # (B, 1, 1)
        w_norm_square = torch.bmm(w, w.permute(0, 2, 1)) # (B, 1, 1)

        # u : (B, D, 1)
        # ((m_wu - w_u) * w) / w_norm_square : (B, 1, D)
        u_hat = u + (((m_wu - w_u) * w) / w_norm_square).permute(0, 2, 1)
        return u_hat

if __name__ == '__main__':
    flow = PlanarFlow(latent_dim=40)
    z = torch.randn(16, 40) 
    z_transformed = flow(z)

    flow.log_abs_det_jacobian(z)