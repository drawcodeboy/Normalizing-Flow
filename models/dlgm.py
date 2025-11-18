import torch
from torch import nn

class DLGM(nn.Module):
    '''
    Deep Latent Gaussian Model (DLGM)
    여기서는 완전한 모델을 구현하지는 않는다.
    왜냐하면, 논문 Page 3의 Section 2.3을 보면, DLGM과 Inference network를 하나의 VAE로 볼 수 있다고 설명했고,
    VAE는 DLGM처럼 잠재변수를 깊게 쌓기보다는 단층적인 구조를 가진다. (여기서 단층, 다층이라는 건 Convolution, Attention 같은 operation이 아니라 Gaussian layer를 의미한다.)
    또한, 논문 Page 8의 Section 6.2에서는 "with 40 latent variables..."라고 표현하는 것을 보면,
    다층 구조의 잠재변수를 언급하기보다는 단층적이기에 저런 표현을 사용했으리라 짐작한다.
    '''
    def __init__(self,
                 latent_dim: int = 40,
                 hidden_dim: int = 400,
                 maxout_window_size: int = 4,
                 output_dim: int = 784):
        super().__init__()

        self.li1 = nn.Linear(latent_dim, hidden_dim*maxout_window_size)
        self.li2 = nn.Linear(hidden_dim, output_dim)

        self.maxout_window_size = maxout_window_size
        self.sigmoid = nn.Sigmoid()

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
        
    def forward(self, z):
        h = self.maxout(self.li1(z))
        x_prime = self.sigmoid(self.li2(h))
        return x_prime