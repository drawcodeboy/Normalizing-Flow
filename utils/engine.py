import torch
import torch.nn.functional as F

import numpy as np
import pdb
import random

from .metrics import get_metrics

@torch.no_grad()
def evaluate(model, dataloader, task_cfg, device):
    model.eval()
    
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