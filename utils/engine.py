import torch
import torch.nn.functional as F

import numpy as np
import pdb
import random

from .metrics import get_metrics

def train_one_epoch(model, dataloader, loss_fn, optimizer, scheduler, task_cfg, device, update_count, max_update_count):
    model.train()
    total_loss = []
    
    for batch_idx, data in enumerate(dataloader, start=1):
        if task_cfg['object'] == 'train_nf':
            optimizer.zero_grad()
            
            x = data
            x = x.to(device)           
            
            x_prime, z_li, mu, log_var = model(x)
            loss = model.free_energy_bound(x, z_li, mu, log_var, x_prime)
            
            total_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        else:
            raise Exception("Check your task_cfg['object'] configuration")
         
        print(f"\rTraining: {100*batch_idx/len(dataloader):.2f}%, Loss: {sum(total_loss)/len(total_loss):.6f}", end="")

        if update_count + batch_idx == max_update_count:
            break
    print()
    
    return sum(total_loss)/len(total_loss), batch_idx

@torch.no_grad()
def evaluate(model, dataloader, task_cfg, device):
    model.eval()
    
    # 랜덤 시드 필요 없다. DCI 논문에서 보면, Q(x)에서 리턴하는 mean을 사용한다고 되어있다.
    # 나 역시 이에 동의한다. z로 샘플링하면서 무작위성을 주면, 그게 평가에 유효하다고 보기에는 적합하지 않다.
    # random_seed = 10
    # random.seed(random_seed)
    # np.random.seed(random_seed)
    # torch.manual_seed(random_seed) # 여기에서는 이 라인이 효과가 있을 가능성이 크다. 이에 대한 근거는 VAE의 Encoder에서 Reparameteriation trick을 CPU 상에서 하고 있기 때문이다.
    # torch.cuda.manual_seed(random_seed)
    # torch.cuda.manual_seed_all(random_seed)
    
    total_x, total_x_prime, total_codes, total_factors = [], [], [], []
    for batch_idx, data in enumerate(dataloader, start=1):
        if task_cfg['object'] == 'test_vae':
            x, label = data
            x = x.to(device)           
            label = label.to(device)

            x_prime, z, mu, log_var = model(x)
            
            total_x.append(x.cpu().numpy())
            total_x_prime.append(x_prime.cpu().numpy())
            total_codes.append(mu.cpu().numpy()) # z 대신 mu
            total_factors.append(label.cpu().numpy())
            
        else:
            raise Exception("Check your task_cfg['object'] configuration")
        
        print(f"\rEvaluate: {100*batch_idx/len(dataloader):.2f}%", end="")
    print()
    
    total_x = np.concatenate(total_x)
    total_x_prime = np.concatenate(total_x_prime)
    total_codes = np.concatenate(total_codes)
    total_factors = np.concatenate(total_factors)
    
    result = get_metrics(total_x, total_x_prime, total_codes, total_factors)
    
    return result