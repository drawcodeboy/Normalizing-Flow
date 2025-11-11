import torch
from torch import nn
import torch.nn.functional as F

from .dlgm import DLGM
from .inference_network import InferenceNetwork
from .flow import PlanarFlow

class NormalizingFlow(nn.Module):
    '''
    
    '''
    def __init__(self,
                 input_dim: int = 784,
                 hidden_dim: int = 400,
                 latent_dim: int = 40,
                 maxout_window_size: int = 4,
                 n_flows: int = 10,
                 flow_type: str = 'planar'):
        super().__init__()

        self.encoder = InferenceNetwork(input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        latent_dim=latent_dim,
                                        maxout_window_size=maxout_window_size)
        self.decoder = DLGM(latent_dim=latent_dim,
                            output_dim=input_dim)
        
        self.n_flows = n_flows
        self.flow = None

        if self.n_flows > 0:
            # Normalizing Flow
            if flow_type == 'planar':
                self.flow = nn.ModuleList([PlanarFlow(latent_dim) for _ in range(n_flows)])
            elif flow_type == 'radial':
                raise Exception("Radial Flow not implemented yet.")
        else:
            # Variational Autoencoder (no flows)
            self.flow = nn.Identity()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)

        z_li = [z]
        if self.n_flows > 0:
            for idx, flow_layer in enumerate(self.flow, start=1):
                z = flow_layer(z)
                if idx == self.n_flows:
                    # 마지막 flow의 z는 저장 안 함
                    break
                z_li.append(z)
        else:
            z = self.flow(z)

        x_prime = self.decoder(z)

        return x_prime, z_li, mu, logvar
    
    def free_energy_bound(self, x, z_li, mu, log_var, x_prime):
        #### VAE Case, Free Energy Bound(-ELBO) (n_flows == 0)
        # 1) Regularization
        first_term = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        
        # 2) Reconstruction
        n = x.size(0)
        second_term = F.mse_loss(x_prime, x, reduction='sum') / n

        loss = first_term + second_term

        # Normalizing Flow Case, Flow-based Free Energy Bound (n_flows > 0)
        if self.n_flows > 0:
            for idx, z in enumerate(z_li, start=0):
                loss += self.flow[idx].log_abs_det_jacobian(z).mean()

        return loss

    @classmethod
    def from_config(cls, cfg):
        return cls(input_dim=cfg['input_dim'],
                   hidden_dim=cfg['hidden_dim'],
                   latent_dim=cfg['latent_dim'],
                   maxout_window_size=cfg['maxout_window_size'],
                   n_flows=cfg['n_flows'])
    
if __name__ == "__main__":
    model = NormalizingFlow(input_dim=784,
                            hidden_dim=400,
                            latent_dim=40,
                            maxout_window_size=4,
                            n_flows=0,
                            flow_type='planar')
    x = torch.randn(16, 784)
    x_prime, z_li, mu, logvar = model(x)
    print("x_prime:", x_prime.shape)
    print("mu:", mu.shape)
    print("logvar:", logvar.shape)
    print("Number of z in z_li:", len(z_li))

    model.free_energy_bound(x, z_li, mu, logvar, x_prime)