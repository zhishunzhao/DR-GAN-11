from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from util.DataAugmentation import FaceIdPoseDataset, RandomCrop, Resize
from PIL import Image
import torch
import numpy as np
import os

class MulPIE(data.Dataset):
    """Dataset class for the MultiPIE dataset."""

    def __init__(self, image_dir, transform, mode, Nd, Np, pose_dict):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.Nd = Nd
        self.Np = Np
        self.pose_dict = pose_dict
        self.train_dataset = []
        self.test_dataset = []
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        dataset1 = ImageFolder(self.image_dir, self.transform)
        for names in dataset1.imgs:
            # lable = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            # id_lable = np.zeros(self.Nd)
            # pose_lable = np.zeros(self.Np)
            name = names[0]
            image_num = int(name.split('/')[-1].split('_')[-1].split('.')[0])
            pose_lable = name.split('/')[-1].split('_')[3]
            if(image_num >= 6 and image_num <= 10 and (pose_lable != '081' or pose_lable != '191')):
                id_lable = int(name.split('/')[-1][0:3])
                pose_lable = self.pose_dict.get(pose_lable)
                # id_lable[id_num] = 1
                # pose_lable[pose_num] = 1
                # print(name)
                # print(id_lable)
                # print(pose_lable)

                # label_trg = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                self.train_dataset.append([name, id_lable, pose_lable])
                self.test_dataset.append([name, id_lable, pose_lable])

        # 该方法是继承torch里面的utils文件夹里面data文件夹里面的Dataset类

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, id_label, pose_lable = dataset[index]
        # filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        # print(image.size)
        return self.transform(image), id_label, pose_lable
        # return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, Nd=200, Np=13, pose_dict={}, image_size=110, batch_size=16, mode='train', num_workers=1,):
    """Build and return a data loader."""
    transform = []
    # if mode == 'train':
    # transform.append(T.RandomHorizontalFlip())
    # transform1.append(T.CenterCrop(178)) 以后会用到裁剪图像
    # to run only once
    transform.append(T.Resize(120))
    transform.append(T.CenterCrop(96))
    # transform.append(RandomCrop((96, 96)))
    transform.append(T.ToTensor())


    # transform.append(T.ToTensor())
    # transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)


    dataset = MulPIE(image_dir, transform, mode, Nd=Nd, Np=Np, pose_dict=pose_dict)

    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=(mode == 'train'),
                                  num_workers=num_workers)
    return data_loader


if(__name__ == '__main__'):
    # im1 = Image.open('1.png')
    # transform = []
    # # if mode == 'train':
    # # transform.append(T.RandomHorizontalFlip())
    # # transform1.append(T.CenterCrop(178)) 以后会用到裁剪图像
    # # to run only once
    # transform.append(T.ToTensor())
    # transform.append(Resize((110, 110)))
    # transform.append(RandomCrop((96, 96)))
    # # transform.append(T.ToTensor())
    # # transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    # transform = T.Compose(transform)
    # im1 = transform(im1)
    # print(im1.shape)
    # print(im1[:,1, 1])
    # pose_dict = {'110': 1, '120': 2, '090': 3, '080': 4, '130': 5, '140': 6, '051': 7, '050': 8, '041': 9,
    #              '190': 10, '200': 11, '010': 12, '240': 13}
    # d = get_loader(image_dir=r'C:\Users\jason\Documents\GitHub\DR-GAN-1\data\session01', Np=13, Nd=200,
    #                                 pose_dict=pose_dict, image_size=110, batch_size=8, mode='train',
    #                                 num_workers=1)
    # for i, batch_data in enumerate(d):
    #     image = torch.FloatTensor(batch_data[0].float())
    #     print(image.size())
    path = r'C:\Users\jason\Documents\GitHub\DR-GAN-1\data\session01\multiview'
    dataset1 = ImageFolder(path, transform=None)
    for names in dataset1.imgs:
        print(names[0])

