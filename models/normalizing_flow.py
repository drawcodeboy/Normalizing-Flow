import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
import pdb

from .dlgm import DLGM
from .inference_network import InferenceNetwork
from .flow import PlanarFlow

class NormalizingFlow(nn.Module):
    '''
    Normalizing Flow
    "Variational Inference with Normalizing Flows" (Rezende & Mohamed, 2015)
    https://arxiv.org/abs/1505.05770
    '''
    def __init__(self,
                 input_dim: int = 784,
                 hidden_dim: int = 400,
                 latent_dim: int = 40,
                 maxout_window_size: int = 4, # maxout activation function
                 n_flows: int = 10,
                 flow_type: str = 'planar'):
        super().__init__()

        self.encoder = InferenceNetwork(input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        latent_dim=latent_dim,
                                        maxout_window_size=maxout_window_size,
                                        n_flows=n_flows,
                                        flow_type=flow_type)
        self.decoder = DLGM(latent_dim=latent_dim,
                            hidden_dim=hidden_dim,
                            maxout_window_size=maxout_window_size,
                            output_dim=input_dim)
        
        self.n_flows = n_flows
        self.flow = None
        self.flow_type = flow_type

        if self.n_flows > 0:
            # Normalizing Flow
            if flow_type == 'planar':
                self.flow = nn.ModuleList([PlanarFlow() for _ in range(n_flows)])
            elif flow_type == 'radial':
                raise Exception("Radial Flow not implemented yet.")
        else:
            # Variational Autoencoder (no flows)
            self.flow = nn.Identity()

        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        if self.n_flows > 0 and self.flow_type == 'planar':
            mu, logvar, u, w, b = self.encoder(x)
        else:
            mu, logvar = self.encoder(x)

        bz = x.size(0)

        z = self.reparameterize(mu, logvar)

        z_li = [z]
        sum_logdet_jacobian = torch.zeros(bz, device=x.device)

        if self.n_flows > 0:
            for flow_idx, flow_layer in enumerate(self.flow, start=0):
                if self.flow_type == 'planar':
                    z, logdet_jacobian = flow_layer(z, u[:, flow_idx, :, :], w[:, flow_idx, :, :], b[:, flow_idx, :, :])
                z_li.append(z)
                sum_logdet_jacobian += logdet_jacobian
        else:
            z = self.flow(z)

        x_prime = self.decoder(z)

        return x_prime, z_li, mu, logvar, sum_logdet_jacobian
    
    def free_energy_bound(self, x, z_li, mu, log_var, x_prime, sum_logdet_jacobian, beta=1.0):
        # Beta is inverse termperature for annealed flow-based free energy bound
        # Expectation by Monte Carlo Sampling (one sample per data point) if term doesn't have analytic form.

        # 1) E_{q_0(z_0)}[ln q_0(z_0)]
        first_term = -Normal(loc=mu, scale=log_var.mul(0.5).exp()).entropy()
        first_term = torch.sum(first_term, dim=-1).mean() # Batch-wise mean

        # 2) Reconstruction
        n = x.size(0)
        second_term = F.binary_cross_entropy(x_prime, x, reduction='sum') / n

        # 3) -E_{q_0(z_0)}[log p(z_k)]
        # if n_flows = 0, -E_{q_0(z_0)}[log p(z_0)]
        z_k = z_li[-1] # if n_flows = 0, z_k == z_0
        third_term = -Normal(loc=torch.zeros_like(z_k), scale=torch.ones_like(z_k)).log_prob(z_k)
        third_term = torch.sum(third_term, dim=-1).mean() # Batch-wise mean

        loss = first_term + beta*(second_term + third_term)

        # 4) Flow correction
        fourth_term = -sum_logdet_jacobian.mean() # Batch-wise mean 

        loss = first_term + beta*(second_term + third_term) + fourth_term

        return loss
    
    def neg_ln_p_x(self, x, samples=200):
        x = x.repeat(samples, 1) # (1, D) -> (samples, D)
        x_prime, z_li, mu, log_var, sum_logdet_jacobian = self.forward(x)

        first_term = -F.binary_cross_entropy(x_prime, x, reduction='none').sum(dim=1)
        
        second_term = Normal(loc=torch.zeros_like(z_li[-1]), scale=torch.ones_like(z_li[-1])).log_prob(z_li[-1]).sum(dim=1)

        third_term = -Normal(loc=mu, scale=log_var.mul(0.5).exp()).log_prob(z_li[0]).sum(dim=1)

        fourth_term = sum_logdet_jacobian
        
        w = first_term + second_term + third_term + fourth_term
        neg_ln_p_x = -torch.logsumexp(w, dim=0) + torch.log(torch.tensor(float(samples)))
        
        return neg_ln_p_x
    
    @classmethod
    def from_config(cls, cfg):
        return cls(input_dim=cfg['input_dim'],
                   hidden_dim=cfg['hidden_dim'],
                   latent_dim=cfg['latent_dim'],
                   maxout_window_size=cfg['maxout_window_size'],
                   n_flows=cfg['n_flows'])