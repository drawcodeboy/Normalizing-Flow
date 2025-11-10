import torch
from torch import nn

class DLGM(nn.Module):
    '''
    Deep Latent Gaussian Model (DLGM)
    '''
    def __init__(self,
                 latent_dim: int = 40,
                 output_dim: int = 784):
        super().__init__()
        
    def forward(self, z):
        h = self.relu(self.li1(z))
        x_recon = self.sigmoid(self.li2(h))
        return x_recon