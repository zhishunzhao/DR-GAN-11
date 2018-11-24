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
from PIL import Image
from torchvision import transforms as T



def Generate_Image(test_dir, pose_code, Nz, G_model, args):
    fixed_noise = torch.FloatTensor(np.random.uniform(-1, 1, (1, Nz)))
    image_number = 1

    if args.cuda:
        G_model.cuda()
        fixed_noise = fixed_noise.cuda()
        pose_code = pose_code.cuda()
    fixed_noise, pose_code = Variable(fixed_noise), Variable(pose_code)
    G_model.eval()
    for root, dirs, files in os.walk(test_dir):
        for f in files:

            file1 = os.path.join(test_dir, f)
            im = Image.open(file1)


            totensor = T.ToTensor()
            im_data = totensor(im).unsqueeze(0)
            print(im_data.shape)
            im_data = Variable(im_data)
            im_data = im_data.cuda()
            # im_data = np.expand_dims(im_data, axis=0)

            generated = G_model(im_data, pose_code, fixed_noise)
            save_generated_image = generated.cpu().data_numpy().transpose(1, 2, 0)
            # 不清楚是否必要
            save_generated_image = np.squeeze(save_generated_image)
            save_generated_image = save_generated_image * 255.0
            save_dir = './test_output'
            filename = os.path.join(save_dir, '{}.jpg'.format(str(image_number)))
            if not os.path.isdir(save_dir): os.makedirs(save_dir)
            print('saving {}'.format(filename))
            misc.imsave(filename, save_generated_image.astype(np.uint8))

            image_number += 1




