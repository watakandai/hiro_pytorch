import numpy as np
import torch

def var(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor

def get_tensor(z):
    if len(z.shape) == 1:
        return var(torch.FloatTensor(z.copy())).unsqueeze(0)
    else:
        return var(torch.FloatTensor(z.copy()))
