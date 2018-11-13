#!/usr/bin/env python
# encoding: utf-8

import torch
import numpy as np
# batch *
def one_hot(batch, depth):
    ones = torch.sparse.torch.eye(depth)
    return ones.index_select(0,batch)

if __name__ == '__main__':
    # b = torch.LongTensor([1, 2, 4])
    # a = one_hot(8, b)
    # print(a)
    tmp = torch.LongTensor(np.random.randint(13, size=6))
    pose_code = one_hot(tmp, 13)
    print(pose_code)