import numpy as np
from einops import rearrange
import torch

time_step_size = 100
patch_size = 5
tensor = torch.arange(100).unsqueeze(0)
print(tensor.shape)

# 1,1,100 -> 1,20,5
patched = rearrange(tensor, 'b (h s1) -> b h s1', s1=patch_size)
print(patched.shape)
print(patched)
