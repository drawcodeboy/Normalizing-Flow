import torch
from torch import nn

class PlanarFlow(nn.Module):
    '''
    Planar Flow (Page 5, Section 4.1)
    '''
    def __init__(self, latent_dim: int):
        super().__init__()
        self.u = nn.Parameter(torch.randn(latent_dim))
        self.w = nn.Parameter(torch.randn(latent_dim))
        self.b = nn.Parameter(torch.randn(1))
        self.h = nn.Tanh()

    def forward(self, z):
        # batch 단위로 내적하려면, 어쩔 수 없이 z가 앞에 와야 함. z @ self.w (수식 그대로 따라가고 싶은데 아쉽.)
        f_z = self.h((z @ self.w) + self.b)
        f_z = z + self.u * f_z.unsqueeze(-1)

        return f_z
    
    def diff_tanh(self, x):
        return 1 - torch.tanh(x) ** 2
    
    def log_abs_det_jacobian(self, z):
        psi_z = self.diff_tanh((z @ self.w) + self.b).unsqueeze(-1) * self.w # (B, D)
        log_abs_det = torch.log(torch.abs(1 + (psi_z @ self.u)))

        return log_abs_det

if __name__ == '__main__':
    flow = PlanarFlow(latent_dim=40)
    z = torch.randn(16, 40) 
    z_transformed = flow(z)

    flow.log_abs_det_jacobian(z)