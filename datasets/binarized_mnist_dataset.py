import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST

class BinarizedMNISTDataset(Dataset):
    def __init__(self,
                 root="data/", # make MNIST directory below data/
                 download=True, # download if there is no data, else pass
                 mode='train'): 
        
        if mode not in ['train', 'test']:
            raise Exception("mode should be 'train' or 'test'")
        
        # Train = 60,000 samples, Test = 10,000 samples
        self.data_li = MNIST(root=root,
                             download=download,
                             train=True if mode=='train' else False)
        
        self.data_li = list(self.data_li)
        
    def __len__(self):
        return len(self.data_li)
    
    def __getitem__(self, idx):
        image, _ = self.data_li[idx]
        
        image = np.array(image.getdata()).reshape(28, 28).astype(np.float32)
        image /= 255.
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        # Binarization
        image = torch.where(image > 0.5, torch.ones_like(image), torch.zeros_like(image))
        
        return image
    
    @classmethod
    def from_config(cls, cfg):
        return cls(root=cfg.get('root', 'data/'),
                   download=cfg.get('download', True),
                   mode=cfg.get('mode', 'train'))

if __name__ == "__main__":
    dataset = BinarizedMNISTDataset(root="data/", download=True, mode='test')
    print(f"Number of samples: {len(dataset)}")
    sample_image = dataset[1000]
    print(f"Sample image shape: {sample_image.shape}")

    import matplotlib.pyplot as plt
    plt.imshow(sample_image.squeeze(0), cmap='gray')
    plt.savefig("datasets/sample_binarized_mnist.png")