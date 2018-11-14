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