from datasets import load_dataset
from models import load_model

from utils import train_one_epoch, save_model_ckpt, save_loss_ckpt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
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
    total_train_loss = []
    total_start_time = int(time.time())
    
    update_count = 0

    model.train()

    while update_count < hp_cfg['max_update_count']:
        print("\n=======================================================")

        loss = None
        for data in train_dl:
            if task_cfg['object'] == 'train_nf':
                optimizer.zero_grad()

                x = data
                x = x.to(device)

                x_prime, z_li, mu, log_var = model(x)
                loss = model.free_energy_bound(x, z_li, mu, log_var, x_prime)

                loss.backward()
                optimizer.step()
            else:
                raise Exception("Check your task_cfg['object'] configuration")
            update_count += 1

            if update_count == hp_cfg['max_update_count']:
                save_model_ckpt(model, save_cfg['name'], update_count, save_cfg['weights_path'])
                break
            elif update_count % 100000 == 0:
                save_model_ckpt(model, save_cfg['name'], update_count, save_cfg['weights_path'])
        print(f"\rParameter update count: {update_count:06d}", end="")

    total_elapsed_time = int(time.time()) - total_start_time
    print(f"<Total Train Time: {total_elapsed_time//60:02d}m {total_elapsed_time%60:02d}s>")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training', parents=[add_args_parser()])
    args = parser.parse_args()

    with open(f'configs/train/{args.config}.yaml') as f:
        cfg = yaml.full_load(f)
    
    main(cfg)