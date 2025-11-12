import torch
import numpy as np
import os

def save_model_ckpt(model, model_name, current_epoch, save_dir):
    ckpt = {}
    ckpt['model'] = model.state_dict()
    
    save_name = f"{model_name}.updates_{current_epoch:03d}.pth"
    
    try:
        torch.save(ckpt, os.path.join(save_dir, save_name))
        print(f"\nSave Model @updates: {current_epoch}")
    except:
        print(f"\nCan\'t Save Model @updates: {current_epoch}")
        
def save_loss_ckpt(model_name, train_loss, save_dir):
    try:
        np.save(os.path.join(save_dir, f'train_loss_{model_name}.npy'), np.array(train_loss))
        print('\nSave Train Loss')
    except:
        print('\nCan\'t Save Train Loss') 