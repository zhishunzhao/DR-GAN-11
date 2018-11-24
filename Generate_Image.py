 #!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
from scipy import misc
import pdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.autograd import Variable
from util.one_hot_for_test import one_hot
from PIL import Image
import torchvision.transforms as T

def Generate_Image(image_folder,Nz, Np, G_model, args):
    """
    Generate_Image with learned Generator

    ### input
    images      : source images
    pose_code   : vector which specify pose to generate image from source image
    Nz          : size of noise vecotr
    G_model     : learned Generator
    args        : options

    ### output
    features    : extracted disentangled features of each image

    """
    image_number = 1
    G_model.cuda()
    G_model.eval()
    folder = image_folder
    os.chdir(folder)
    pose_code = torch.zeros((1, Np))
    pose_code[5] = 1
    pose_code = Variable(pose_code)
    pose_code = pose_code.cuda()
    fixed_noise = torch.FloatTensor(np.random.uniform(-1, 1, (1, Nz)))
    fixed_noise = Variable(fixed_noise)
    fixed_noise = fixed_noise.cuda()

    for f in os.listdir(os.getcwd()):
        im = Image.open(f)
        im = T.ToTensor()(im).unsqueeze(0)
        im = im.cuda()
        im = Variable(im)
        generated = G_model(im, pose_code, fixed_noise)
        save_generated_image = generated.cpu().data_numpy().transpose(1, 2, 0)
        # 不清楚是否必要
        save_generated_image = np.squeeze(save_generated_image)
        save_generated_image = save_generated_image * 255.0
        save_dir = '/home/home_data/jason/DR-GAN-11/test_output'
        filename = os.path.join(save_dir, '{}.jpg'.format(str(image_number)))
        if not os.path.isdir(save_dir): os.makedirs(save_dir)
        print('saving {}'.format(filename))
        misc.imsave(filename, save_generated_image.astype(np.uint8))

        image_number += 1

