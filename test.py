from datasets import load_dataset
from models import load_model

import torch
from torch import nn
import numpy as np
import argparse, time, os, sys, yaml

def add_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str)

    return parser
        
def main(cfg):
    print(f"=====================[{cfg['title']}]=====================")

    # Device Setting
    device = None
    if cfg['device'] != 'cpu' and torch.cuda.is_available():
        device = cfg['device']
    else: 
        device = 'cpu'
    print(f"device: {device}")

    # Load Settings
    hp_cfg = cfg['hyperparameters']
    task_cfg = cfg['task']
    save_cfg = cfg['save']

    # Load Dataset
    test_data_cfg = cfg['data']['test']
    test_ds = load_dataset(test_data_cfg)
    test_dl = torch.utils.data.DataLoader(test_ds,
                                          shuffle=False,
                                          batch_size=hp_cfg['batch_size'],
                                          drop_last=False)
    print(f"Load Test Dataset {test_data_cfg['name']}")
            
    # Load Model
    model_cfg = cfg['model']
    print(model_cfg['name'])
    model = load_model(model_cfg).to(device)
    ckpt = torch.load(os.path.join(save_cfg['weights_path'], save_cfg['weights_filename']),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    
    if cfg['parallel'] == True:
        model = nn.DataParallel(model)

    start_time = int(time.time())
    
    update_count = 0
    model.eval()

    feb_li = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_dl, start=1):

            if task_cfg['object'] == 'test_nf':

                x = data
                x = x.to(device)

                x_prime, z_li, mu, log_var, sum_logdet_jacobian = model(x)
                feb = model.free_energy_bound(x, z_li, mu, log_var, x_prime, sum_logdet_jacobian, beta=1.0)
                feb_li.append(feb.item())
            else:
                raise Exception("Check your task_cfg['object'] configuration")
            
            update_count += 1
            elapsed_time = int(time.time()) - start_time
            print(f"\r[Time: {elapsed_time//60:02d}m {elapsed_time%60:02d}s] ({100*batch_idx/len(test_dl)}%)", end="")
    
    print()
    avg_feb = sum(feb_li)/len(feb_li)
    print(f"Average Free Energy Bound: {avg_feb:.6f}")

    test_dl = torch.utils.data.DataLoader(test_ds,
                                          shuffle=False,
                                          batch_size=1,
                                          drop_last=False)

    neg_ln_p_x_li = []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_dl, start=1):

            if task_cfg['object'] == 'test_nf':

                x = data
                x = x.to(device)

                neg_ln_p_x = model.neg_ln_p_x(x, samples=200)
                neg_ln_p_x_li.append(neg_ln_p_x.item())
            else:
                raise Exception("Check your task_cfg['object'] configuration")
            
            elapsed_time = int(time.time()) - start_time
            print(f"\r[Time: {elapsed_time//60:02d}m {elapsed_time%60:02d}s] ({100*batch_idx/len(test_dl)}%)", end="")

    print()
    avg_neg_ln_p_x = sum(neg_ln_p_x_li)/len(neg_ln_p_x_li)
    print(f"Average Negative ln p(x): {avg_neg_ln_p_x:.6f}")

    print(f"KL Divergence between q(z|x) and p(z|x): {avg_feb - avg_neg_ln_p_x:.6f}") # -ELBO + ln p(x)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test', parents=[add_args_parser()])
    args = parser.parse_args()

    with open(f'configs/test/{args.config}.yaml') as f:
        cfg = yaml.full_load(f)
    
    main(cfg)