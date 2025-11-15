import os, sys
sys.path.append(os.getcwd())

import torch
from models.flow import PlanarFlow
flow=PlanarFlow(40)

for i in range(5):
    z=torch.randn(100,40)
    ld=flow.log_abs_det_jacobian(z)
    print(f"iter {i}: mean={ld.mean().item():.6f}, std={ld.std().item():.6f}, min={ld.min().item():.6f}, max={ld.max().item():.6f}")


flow.u.data*=0.01
flow.w.data*=0.01
z=torch.randn(100,40)
ld=flow.log_abs_det_jacobian(z)
print(f"small init: mean={ld.mean().item():.6f}, std={ld.std().item():.6f}, min={ld.min().item():.6f}, max={ld.max().item():.6f}")