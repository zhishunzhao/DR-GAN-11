from main import DataLoader
from util.Multipie_loader import get_loader


Nd, Np, Nz, channel_num = DataLoader()
pose_dict = {'110': 1, '120': 2, '090': 3, '080': 4, '130': 5, '140': 6, '051': 7, '050': 8, '041': 9,
                         '190': 10, '200': 11, '010': 12, '240': 13}
dataloader = get_loader(image_dir=r'C:\Users\jason\Documents\GitHub\DR-GAN-1\data\session01', Np=13, Nd=200,
                                    pose_dict=pose_dict, image_size=110, batch_size=8, mode='train',
                                    num_workers=1)
print('Nd: ', Nd)
print('Np: ', Np)
print('Nz: ', Nz)
print('channel_num: ', channel_num)
print(dataloader)
"""
Traceback (most recent call last):
  File "main.py", line 135, in <module>
    train_single_DRGAN(dataloader, Nd, Np, Nz, D, G, args)
  File "/home/manager/jason/DR-GAN-11/train_single_DRGAN.py", line 117, in train_single_DRGAN
    batch_id_label, batch_pose_label, batch_ones_label, batch_zeros_label, epoch, steps, Nd, args)
  File "/home/manager/jason/DR-GAN-11/train_single_DRGAN.py", line 147, in Learn_D
    L_id    = loss_criterion(real_output[:, :Nd], batch_id_label)
  File "/home/manager/anaconda3/lib/python3.6/site-packages/torch/nn/modules/module.py", line 491, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/manager/anaconda3/lib/python3.6/site-packages/torch/nn/modules/loss.py", line 759, in forward
    self.ignore_index, self.reduce)
  File "/home/manager/anaconda3/lib/python3.6/site-packages/torch/nn/functional.py", line 1442, in cross_entropy
    return nll_loss(log_softmax(input, 1), target, weight, size_average, ignore_index, reduce)
  File "/home/manager/anaconda3/lib/python3.6/site-packages/torch/nn/functional.py", line 1332, in nll_loss
    return torch._C._nn.nll_loss(input, target, weight, size_average, ignore_index, reduce)
RuntimeError: multi-target not supported at /pytorch/aten/src/THCUNN/generic/ClassNLLCriterion.cu:16
"""