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
                z_li.append(z)
        else:
            z = self.flow(z)

        x_prime = self.decoder(z)

        return x_prime, z_li, mu, logvar
    
    def free_energy_bound(self, x, z_li, mu, log_var, x_prime, beta=1.0):
        # Beta is inverse termperature for annealed flow-based free energy bound

        # VAE Case, Free Energy Bound(-ELBO) (n_flows == 0)
        if self.n_flows == 0:
            # 1) Regularization
            first_term = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
            
            # 2) Reconstruction
            n = x.size(0)
            second_term = F.binary_cross_entropy(x_prime, x, reduction='sum') / n

            loss = first_term + second_term

        # Normalizing Flow Case, Flow-based Free Energy Bound (n_flows > 0)
        elif self.n_flows > 0:
            if beta is None: beta = 1.0

            # Expectation by Monte Carlo Sampling (one sample per data point)

            # 1) E_{q_0(z_0)}[ln q_0(z_0)]
            first_term = -Normal(loc=mu, scale=log_var.mul(0.5).exp()).entropy()
            first_term = torch.sum(first_term, dim=-1).mean() # Batch-wise mean

            # 2) Reconstruction
            n = x.size(0)
            second_term = F.binary_cross_entropy(x_prime, x, reduction='sum') / n

            # 3) -E_{q_0(z_0)}[log p(z_K)]
            z_K = z_li[-1]
            third_term = -Normal(loc=torch.zeros_like(z_K), scale=torch.ones_like(z_K)).log_prob(z_K)
            third_term = torch.sum(third_term, dim=-1).mean() # Batch-wise mean

            # 4) Flow correction
            fourth_term = 0.
            for idx, z in enumerate(z_li, start=0):
                if idx == (len(z_li)-1): break
                fourth_term -= self.flow[idx].log_abs_det_jacobian(z).mean() # batch-wise mean

            loss = first_term + beta*(second_term + third_term) + fourth_term

        return loss
    
    def neg_ln_p_x(self, x, samples=200):
        x = x.repeat(samples, 1) # (1, D) -> (samples, D)
        x_prime, z_li, mu, log_var = self.forward(x)

        if self.n_flows == 0:
            first_term = -F.binary_cross_entropy(x_prime, x, reduction='none').sum(dim=1)
            
            second_term = Normal(loc=torch.zeros_like(z_li[-1]), scale=torch.ones_like(z_li[-1])).log_prob(z_li[-1]).sum(dim=1)

            third_term = -Normal(loc=mu, scale=log_var.mul(0.5).exp()).log_prob(z_li[0]).sum(dim=1)
            
            w = first_term + second_term + third_term
            neg_ln_p_x = -torch.logsumexp(w, dim=0) + torch.log(torch.tensor(float(samples)))

        elif self.n_flows > 0:
            first_term = -F.binary_cross_entropy(x_prime, x, reduction='none').sum(dim=1)
            
            second_term = Normal(loc=torch.zeros_like(z_li[-1]), scale=torch.ones_like(z_li[-1])).log_prob(z_li[-1]).sum(dim=1)

            third_term = -Normal(loc=mu, scale=log_var.mul(0.5).exp()).log_prob(z_li[0]).sum(dim=1)

            fourth_term = torch.zeros_like(first_term)
            for idx, z in enumerate(z_li, start=0):
                if idx == (len(z_li)-1): break
                fourth_term += self.flow[idx].log_abs_det_jacobian(z)
            
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
    
if __name__ == "__main__":
    model = NormalizingFlow(input_dim=784,
                            hidden_dim=400,
                            latent_dim=40,
                            maxout_window_size=4,
                            n_flows=40,
                            flow_type='planar')
    x = torch.ones(1, 784)
    x_prime, z_li, mu, logvar = model(x)
    print("x_prime:", x_prime.shape)
    print("mu:", mu.shape)
    print("logvar:", logvar.shape)
    print("Number of z in z_li:", len(z_li))

    model.free_energy_bound(x, z_li, mu, logvar, x_prime)
    print(model.neg_ln_p_x(x, samples=200).shape)