import torch
from torch import nn
from einops import rearrange

class InferenceNetwork(nn.Module):
    '''
    Inference network is Encoder in VAE.
    '''
    def __init__(self,
                 input_dim: int = 784,
                 hidden_dim: int = 400,
                 latent_dim: int = 40,
                 maxout_window_size: int = 4,
                 n_flows: int = 0):
        super().__init__()
        
        self.li1 = nn.Linear(input_dim, hidden_dim*maxout_window_size)
        self.li2_mu = nn.Linear(hidden_dim, latent_dim)
        self.li2_logvar = nn.Linear(hidden_dim, latent_dim)

        self.maxout_window_size = maxout_window_size

        self.n_flows = n_flows

    def maxout(self, x):
        '''
        Maxout activation function.
        In page 7 of the original paper (left-column)
        '''
        shape = x.size()

        # (B, D*window) -> (B, D, window)
        new_shape = shape[:-1] + (shape[-1] // self.maxout_window_size, self.maxout_window_size)
        x = x.view(new_shape)
        x, _ = torch.max(x, dim=-1)
        return x
    
    def forward(self, x):
        h = self.maxout(self.li1(x))
        mu = self.li2_mu(h)
        logvar = self.li2_logvar(h)
        return mu, logvar

if __name__ == "__main__":
    model = InferenceNetwork()
    x = torch.randn(16, 784)
    h = model.maxout(model.li1(x))
    print("h:", h.shape)

    mu, logvar = model(x)
    print("mu:", mu.shape)
    print("logvar:", logvar.shape)