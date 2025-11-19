from datasets import load_dataset
from models import load_model

from utils import save_model_ckpt, save_loss_ckpt

import torch
from torch import nn, optim
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

    # Hyperparameter Settings
    hp_cfg = cfg['hyperparameters']

    # Load Dataset
    train_data_cfg = cfg['data']['train']
    train_ds = load_dataset(train_data_cfg)
    train_dl = torch.utils.data.DataLoader(train_ds,
                                           shuffle=True,
                                           batch_size=hp_cfg['batch_size'],
                                           drop_last=True)
    print(f"Load Train Dataset {train_data_cfg['name']}")
            
    # Load Model
    model_cfg = cfg['model']
    print(model_cfg['name'])
    model = load_model(model_cfg).to(device)
    
    if cfg['parallel'] == True:
        model = nn.DataParallel(model)

    
    
    # Optimizer
    optimizer = None
    if hp_cfg['optim'] == "RMSprop":
        # 1 x 10^-5, momentum=0.9
        optimizer = optim.RMSprop(model.parameters(), lr=hp_cfg['lr'], momentum=hp_cfg['momentum'])
    elif hp_cfg['optim'] == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=hp_cfg['lr'])
    
    task_cfg = cfg['task']
    save_cfg = cfg['save']

    # Training loss
    start_time = int(time.time())
    
    update_count = 0
    model.train()

    while update_count <= hp_cfg['max_update_count']:
        loss = None
        break_flag = False
        for data in train_dl:

            if task_cfg['object'] == 'train_nf':
                beta = min(1.0, 0.01 + update_count / 10000.0) if hp_cfg['use_inv_temperature'] else 1.0

                optimizer.zero_grad()

                x = data
                x = x.to(device)

                x_prime, z_li, mu, log_var, sum_logdet_jacobian = model(x)
                loss = model.free_energy_bound(x, z_li, mu, log_var, x_prime, sum_logdet_jacobian, beta=beta)

                loss.backward()
                optimizer.step()
            else:
                raise Exception("Check your task_cfg['object'] configuration")

            if update_count == hp_cfg['max_update_count']:
                save_model_ckpt(model, save_cfg['name'], update_count, save_cfg['weights_path'])
                break_flag = True
                break
            elif (update_count % 100000 == 0) and (update_count != 0):
                save_model_ckpt(model, save_cfg['name'], update_count, save_cfg['weights_path'])
            
            update_count += 1
            elapsed_time = int(time.time()) - start_time
            print(f"\r[Time: {elapsed_time//60:02d}m {elapsed_time%60:02d}s] Parameter update count: {update_count:06d} / Free Energy Bound: {loss:.6f} / beta: {beta:.4f}", end="")

        if break_flag: break
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training', parents=[add_args_parser()])
    args = parser.parse_args()

    with open(f'configs/train/{args.config}.yaml') as f:
        cfg = yaml.full_load(f)
    
    main(cfg)