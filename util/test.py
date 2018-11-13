from PIL import Image
import numpy as np
import os
import torch

# im = Image.open('timg.jpg')
# im1 = Image.open('1.png')
# im2 = Image.open('2.png')
# im.show()
# width, height = im.size
#
# data = im.getdata()
# data = np.matrix(data,dtype='float')/255.0
# new_data = np.reshape(data, (700, 990, 3))
# print(new_data.shape)
# print(im.getpixel((0, 0)))
# print(im.mode)
# print(width, height)
# data = im.getdata()
# print(data.size)
# img = np.array(im)
# im1 = np.array(im1)
# im2 = np.array(im2)
# im3 = np.stack((im1, im2))
# im4 = np.stack(im3, im1)
# im4 = np.expand_dims(im1, axis=0)
# im5 = np.concatenate((im3, im4), axis = 0)
# print(im1.shape)
# print(im2.shape)
# print(im3.shape)
# print(im4.shape)
# print(img.shape)
# print(im5.shape)
# image = np.array(im)
# image = image.reshape((3, 990, 700))
#
# print(image.shape)
# image = image.transpose(2, 0, 1)
# print(image.shape)
x = np.random.randn(8, 320)
x = torch.from_numpy(x)
print(x.shape)
pose = np.random.randn(8, 14)
pose = torch.from_numpy(pose)
noise = np.random.randn(8, 50)
noise = torch.from_numpy(noise)
x = torch.cat([x, pose, noise], 1)
print(x.shape)