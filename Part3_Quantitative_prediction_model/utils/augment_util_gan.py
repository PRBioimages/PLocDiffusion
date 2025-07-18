import sys
sys.path.insert(0, '..')

import math
import numpy as np
import cv2
import torch
from config.config_ import *
from run.gan_test import ganmodel_
from utils.tool import label_to_tensor
label_rev_anno = {
    0: 'Cy',
    1: 'Np',
    2: 'Mi',
    3: 'Nu',
    4: 'PM',
    5: 'CyNp',
    6: 'CyPM',
    7 : 'NuMi',
    8 : 'NuNu'
}
ganmodel = ganmodel_()


def train_gan_augment(image,label_5,p):

    augment_func_list = [
        lambda image,label_5: (image,label_5,0),
        augment_label2,
        augment_label3,
        augment_label4,
    ]
    c = np.random.choice(len(augment_func_list),p=p)
    image, label, flag = augment_func_list[c](image,label_5)
    return image, label, flag


def augment_label_(image,label_5,comb):
    if comb == 'Nu':
        label_2 = np.array([[1,0]]).astype('float32')
    elif comb == 'PM':
        label_2 = np.array([[0, 1]]).astype('float32')
    img = ganmodel.gan_test_(image,label_5,label_2,comb)
    label = label2_2_5(label_2,comb)
    return img, label,1

def augment_label1(image,label_5):
    comb = label_5to2(label_5)
    label_2 = np.array([[0,1]]).astype('float32')

    img = ganmodel.gan_test_(image,label_2,comb)
    label = label2_2_5(label_2,comb)
    return img, label,1

def augment_label2(image,label_5):
    comb = label_5to2(label_5)

    label_2 = np.array([[0.25,0.75]]).astype('float32')
    img = ganmodel.gan_test_(image,label_2,comb)
    label = label2_2_5(label_2, comb)
    return img, label,1

def augment_label3(image,label_5):
    comb = label_5to2(label_5)
    label_2 = np.array([[0.5,0.5]]).astype('float32')
    img = ganmodel.gan_test_(image,label_2,comb)
    label = label2_2_5(label_2, comb)
    return img, label,1

def augment_label4(image,label_5):
    comb = label_5to2(label_5)
    label_2 = np.array([[0.75,0.25]]).astype('float32')
    img = ganmodel.gan_test_(image,label_2,comb)
    label = label2_2_5(label_2, comb)
    return img, label,1

def augment_label5(image,label_5):
    comb = label_5to2(label_5)
    label_2 = np.array([[1,0]]).astype('float32')
    img = ganmodel.gan_test_(image,label_2,comb)
    label = label2_2_5(label_2, comb)
    return img, label,1

def label_5to2(label):
    comb = 'None'
    binlabel = label>0
    for i in baselabel.keys():
        if (baselabel[i] == binlabel).all():
            comb = label_rev_anno[i]
            break
    if comb == 'None':
        for i in baselabel.keys():
            b = baselabel[i]
            a = baselabel[i]
            if (baselabel[i][0] == binlabel).all():
                comb = label_rev_anno[i]
                break

    return comb

def label2_2_5(label2,comb):
    label5 = np.array(baselabel[label_anno[comb]]).astype('float32')
    label2 = label2.squeeze()
    if comb == 'CyNp':
        label5[0] = label2[0]
        label5[1] = label2[1]
    elif comb == 'CyPM':
        label5[0] = label2[0]
        label5[4] = label2[1]
    elif comb == 'NuMi':
        label5[1] = label2[1]
        label5[2] = label2[0]
    elif comb == 'NuNu':
        label5[1] = label2[1]
        label5[3] = label2[0]
    return label5

