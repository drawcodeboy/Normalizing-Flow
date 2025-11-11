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
    
    min_loss = 1e10
    update_count = 0
    
    for current_epoch in range(1, hp_cfg['epochs']+1):
        print("=======================================================")
        print(f"Epoch: [{current_epoch:03d}/{hp_cfg['epochs']:03d}]\n")
        print(f"Parameter update count: {update_count:06d}\n")
        
        # Training One Epoch
        start_time = int(time.time())
        train_loss, update_count_per_epoch = train_one_epoch(model, train_dl, None, optimizer, None, task_cfg, device, update_count, hp_cfg['max_update_count'])
        elapsed_time = int(time.time() - start_time)
        print(f"Train Time: {elapsed_time//60:02d}m {elapsed_time%60:02d}s\n")

        if train_loss < min_loss:
            min_loss = train_loss
            save_model_ckpt(model, save_cfg['name'], current_epoch, save_cfg['weights_path'])

        total_train_loss.append(train_loss)
        save_loss_ckpt(save_cfg['name'], total_train_loss, save_cfg['loss_path'])

        update_count += update_count_per_epoch
        if update_count == hp_cfg['max_update_count']:
            print(f"Reached max update count: {hp_cfg['max_update_count']}. Stopping training.")
            break

    total_elapsed_time = int(time.time()) - total_start_time
    print(f"<Total Train Time: {total_elapsed_time//60:02d}m {total_elapsed_time%60:02d}s>")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training', parents=[add_args_parser()])
    args = parser.parse_args()

    with open(f'configs/train/{args.config}.yaml') as f:
        cfg = yaml.full_load(f)
    
    main(cfg)