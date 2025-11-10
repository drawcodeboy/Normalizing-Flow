from .binarized_mnist_dataset import BinarizedMNISTDataset

def load_dataset(cfg):
    if cfg['name'] == 'binarized_mnist':
        return BinarizedMNISTDataset.from_config(cfg)
    else:
        raise Exception(f"Dataset {cfg['name']} not supported.")