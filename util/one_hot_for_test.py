import torch
import numpy as np
# batch *
def one_hot(batch_size, Np):
    zeros = torch.zeros((batch_size, Np))
    for i in range(batch_size):
        zeros[i][5] = 1
    return zeros
