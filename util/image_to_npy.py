from PIL import Image
import numpy as np
import os

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
# ROOT_DIR = 'data'
# Angel_dict = {'11_0':'0', '12_0':'15'}
#
#
#
#
# for sess_num in range(1, 5):
#     cur_dir = os.path.join(ROOT_DIR, 'session0', str(sess_num))
#     os.chdir(cur_dir)
#     folder_num = 0
#     for p in os.listdir(os.getcwd()):
#         if os.path.isdir(p):
#             folder_num += 1
#     for id in range(1, folder_num + 1):
#         cur_dir_id = os.path.join(cur_dir, str(id).zfill(3))
#         cur_dir_id_t = os.path.join(cur_dir_id, '01')
#         os.chdir(cur_dir_id_t)
#         folder_num_1 = 0
#         for p in os.listdir(os.getcwd()):
#             if os.path.isdir(p):
#                 folder_num_1 += 1


ROOT_DIR = r'C:\Users\jason\Documents\GitHub\DR-GAN\data'

session_dict = {1:250, 2:292, 3:346, 4:346}
POSE_DICT = {'110':1, '120':2, '090':3, '080':4, '130':5, '140':6, '051':7, '050':8, '041':9, '190':10, '200':11, '010':12, '240':13}
image_num = 0
id_label_array = []
pose_label_array = []

for sess in range(1, 2):
    s_num = 'session' + str(sess).zfill(2)
    current_dir = os.path.join(ROOT_DIR, s_num, 'multiview')
    # for id in range(1, session_dict.get(sess) + 1):
    for id in range(1, 2):
        current_dir_t = os.path.join(current_dir, str(id).zfill(3), '01')
        os.chdir(current_dir_t)
        for p in os.listdir(os.getcwd()):
           if(os.path.isdir(p) and p != '08_1' and p != '19_1'):
                current_dir_t_pose = os.path.join(current_dir_t, p)
                os.chdir(current_dir_t_pose)
                for im in os.listdir(os.getcwd()):
                    id_label = im.split('_')[0]
                    id_label_array.append(id_label)

                    pose_label = POSE_DICT.get(im.split('_')[3])
                    pose_label_array.append(pose_label)
                    # Height * Width * CHANNEL
                    image = Image.open(im)
                    image = np.array(image, dtype=float)
                    # image = image.transpose(2, 0, 1)
                    if(image_num == 0):
                        res = image
                        image_num += 1
                    elif(image_num == 1):
                        res = np.stack((res, image))
                        image_num += 1
                    else:
                        image = np.expand_dims(image, axis=0)
                        res = np.concatenate((res, image), axis=0)
                        image_num += 1
                    print('Possessing {} images'.format(image_num))
os.chdir(ROOT_DIR)
res = res.transpose(0, 3, 1, 2)
print(res.shape)
np.save('images.npy', res)
np_id = np.array(id_label_array, dtype=int)
np_pose = np.array(pose_label_array, dtype=int)
np.save('ids.npy', np_id)
np.save('yaws.npy', np_pose)






