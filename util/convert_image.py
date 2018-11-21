#!/usr/bin/env python
# encoding: utf-8

import numpy as np

## convert B x C x H x W BGR [-1,1] to B x H x W x C RGB [0,255]

def convert_image(data):
    if len(data.shape)==4:
        img = data.transpose(0, 2, 3, 1)
        # img = img / 2.0
<<<<<<< HEAD
        img = data * 255.
        # img = img[:,:,:,[2,1,0]]
=======
        img = img * 255.
        img = img[:,:,:,[2,1,0]]
>>>>>>> parent of 6e1cd97... test

    else:
        img = data.transpose(1, 2, 0)
        # img = img / 2.0
<<<<<<< HEAD
        img = data * 255.
        # img = img[:,:,[2,1,0]]
=======
        img = img * 255.
        img = img[:,:,[2,1,0]]
>>>>>>> parent of 6e1cd97... test

    return img.astype(np.uint8)
