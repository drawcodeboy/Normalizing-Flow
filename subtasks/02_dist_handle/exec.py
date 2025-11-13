import torch
from torch.distributions import Normal

def main():
    '''
    first_term = -Normal(loc=mu, scale=log_var.mul(0.5).exp()).entropy()
    third_term = -Normal(loc=torch.zeros_like(z_K), scale=torch.ones_like(z_K)).log_prob(z_K)
    # Free Energy bound를 구현하면서 들었던 의문점
    # 지금 mu, log_var, z_K의 shape이 (batch_size, latent_dim)인데
    # 샘플링하면 (batch_size, latent_dim) shape의 샘플이 나옴
    # 즉, latent_dim shape이 나오게끔 하는 게 목적 아닌가 왜 이렇게 하지
    # 이러면 그냥 32*8 shape의 샘플이 모두 각각의 normal distribution으로부터 샘플링된 거 아닌가
    '''
    mu = torch.zeros((4, 8)) + 2
    sigma = torch.ones((4, 8)) + 5
    dist = Normal(loc=mu, scale=sigma)

    sample = dist.sample()
    print(sample.shape) # (B, latent_dim)

    print(dist.entropy().shape) # (B, latent_dim)
    print(dist.entropy().sum(dim=-1))

if __name__ == '__main__':
    main()