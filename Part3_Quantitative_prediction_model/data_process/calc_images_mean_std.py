import os
from utils.common_util import *
from config.config_ import *
import numpy as np
from utils.augment_util_ import train_vit_transform
import cv2
# def get_img_mean_std(imgs, img_mean, img_std):
#     for i in range(len(imgs)):
#         img = imgs[i,:,:]
#         # img = train_vit_transform(img * 255) / 255
#         m = img.mean()
#         s = img.std()
#         img_mean.append(m)
#         img_std.append(s)
#     return img_mean, img_std

def get_img_mean_std(img, img_mean, img_std):
    img = train_vit_transform(img * 255) / 255
    m = img.mean()
    s = img.std()
    img_mean.append(m)
    img_std.append(s)
    return img_mean, img_std

if __name__ == "__main__":
    if 'lymo' != dataset.lower():
        print('%s: calling main function ... ' % os.path.basename(__file__))
        color = ['blue', 'green', 'yellow', 'red']
        comb = 'NuNu'
        datasetsfile = TrainImages_Classfication_h5
        h5file = h5py.File(datasetsfile, 'r')
        for i in range(len(color)):
            img_mean = []
            img_std = []
            for icate in h5file.keys():
                train_images = h5file.get(icate).value
                img_mean, img_std = get_img_mean_std(train_images[:,:,:,i], img_mean, img_std)

            print(color[i], np.around(np.mean(img_mean), decimals=6), np.around(np.mean(img_std), decimals=6))
    else:
        print('%s: calling main function ... ' % os.path.basename(__file__))
        for color in ['red', 'green']:
            if color == 'red':
                ch = 0
            else:
                ch = 1
            img_mean = []
            img_std = []
            img_dir = TrainImgs_basedir
            for img_list in os.listdir(img_dir):
                img = cv2.imread(opj(img_dir,img_list))[:,:,ch] / 255
                img_mean, img_std = get_img_mean_std(img, img_mean, img_std)
            img_dir = ValImgs_basedir
            for img_list in os.listdir(img_dir):
                img = cv2.imread(opj(img_dir, img_list))[:, :, ch] / 255
                img_mean, img_std = get_img_mean_std(img, img_mean, img_std)
            print(color, np.around(np.mean(img_mean), decimals=6), np.around(np.mean(img_std), decimals=6))

    print('\nsuccess!')