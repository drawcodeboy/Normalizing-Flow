from .normalizing_flow import NormalizingFlow

def load_model(cfg):
    if cfg['name'] == 'normalizing_flow':
        return NormalizingFlow.from_config(cfg)
    else:
        raise Exception(f"Unknown model name: {cfg['name']}")