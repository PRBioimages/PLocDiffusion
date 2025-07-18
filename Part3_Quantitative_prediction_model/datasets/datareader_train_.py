from torch.utils.data.dataset import Dataset
from config.config_ import *
from utils.common_util import *
from utils.tool import *
import cv2
import pickle
import pandas as pd

class data_reader(Dataset):
    def __init__(self,net,Images_Norm_params,is_Train=True,is_Syn=True,transform=None,ind=None):

        if not ope(TrainLabels_Classfication_h5):
            raise (FileNotFoundError("The file {} is not found!".format(TrainLabels_Classfication_h5)))
        if not ope(ValLabels_Classfication_h5):
            raise (FileNotFoundError("The file {} is not found!".format(ValLabels_Classfication_h5)))
        if not ope(TrainImgs_basedir):
            raise (FileNotFoundError("The file {} is not found!".format(TrainImgs_basedir)))
        if not ope(ValImgs_basedir):
            raise (FileNotFoundError("The file {} is not found!".format(ValImgs_basedir)))

        self.Images_Norm_params = Images_Norm_params
        self.is_Train = is_Train
        self.is_Syn = is_Syn
        self.transform = transform
        self.net = net
        if FOLD:
            if not self.is_Syn:
                self.ind =ind
                self.t_imgbasedir = TrainImgs_basedir
                self.t_labels = self.read_h5_classfication(TrainLabels_Classfication_h5)
                self.v_imgbasedir = ValImgs_basedir
                self.v_labels = self.read_h5_classfication(ValLabels_Classfication_h5)
                self.num = len(self.ind)
            else:
                self.imgbasedir = ValSynImgs_basedir_
                self.labels = self.read_h5_classfication(ValSynLabels_Classfication_h5_)
                self.num = len(self.labels)
        else:
            if is_Train:
                self.imgbasedir = TrainImgs_basedir
                self.labels = self.read_h5_classfication(TrainLabels_Classfication_h5)
            else:
                if is_Syn:
                    self.imgbasedir = ValSynImgs_basedir_
                    self.labels = self.read_h5_classfication(ValSynLabels_Classfication_h5_)
                else:
                    self.imgbasedir = ValImgs_basedir
                    self.labels = self.read_h5_classfication(ValLabels_Classfication_h5)
            self.num = len(self.labels)

    def read_h5_classfication(self,datasetsfile):
        h5file = h5py.File(datasetsfile, 'r')
        data = h5file['labels'][:]
        return data

    def read_imgs_classfication(self,basedir,idx,label=None):
        BGYimg = cv2.imread(opj(basedir,'BGYImg'+str(idx)+'.png'))
        img = np.zeros([256,256,3],dtype='float32')
        img[:, :, 0] = BGYimg[:, :, 0]
        img[:, :, 1] = BGYimg[:, :, 1]
        img[:, :, 2] = BGYimg[:, :, 2]
        img = img / 255
        return img


    def changelabel(self, label):
        idx = np.where(label == 1)
        if len(label[idx]) == 2:
            comb = self.label_5to2(label)
            if comb == 'None':
                raise ValueError('comb can not be None!')
            idx = np.array(idx).squeeze()
            chantmplabel = [0.5, 0.5]
            for n in range(2):
                label[idx[n]] = chantmplabel[n]
        return label

    def label_5to2(self,label):
        label_rev_anno = {
            0: 'Cy',
            1: 'Np',
            2: 'Mi',
            3: 'Nu',
            4: 'PM',
            5: 'CyNp',
            6: 'CyPM',
            7: 'NuMi',
            8: 'NuNu'
        }
        comb = 'None'
        binlabel = label > 0
        # print(binlabel)
        for i in baselabel.keys():
            if (baselabel[i] == binlabel).all():
                comb = label_rev_anno[i]
                break
        return comb

    def find_imgs(self, index):
        flag = 0
        if FOLD:
            if not self.is_Syn:
                if index >= len(self.t_labels):
                    imgbasedir = self.v_imgbasedir
                    idx = index-len(self.t_labels)
                    label = self.v_labels
                else:
                    imgbasedir = self.t_imgbasedir
                    idx = index
                    label = self.t_labels
            else:
                label = self.labels
                idx = index
                imgbasedir = self.imgbasedir
            if 'diffusion' in self.net.lower():
                curimg = self.read_imgs_classfication(imgbasedir, idx, label)
            else:
                curimg = self.read_imgs_classfication(imgbasedir,idx,label)
                curlabel = label[idx].astype('float32')

        else:
            if 'diffusion' in self.net.lower():
                if self.transform is not None:
                    curimg = self.read_imgs_classfication(self.imgbasedir,index)
                else:
                    curimg = self.read_imgs_classfication(self.imgbasedir, index)
                    curlabel = self.labels[index].astype('float32')
            else:
                curimg = self.read_imgs_classfication(self.imgbasedir,index)
                curlabel = self.labels[index].astype('float32')

        curimg = self.extract_imgs(curimg)
        if not self.is_Syn:
            if 'diffusion' in self.net.lower():
                if not flag:
                    curlabel = self.changelabel(curlabel)
            else:
                curlabel = self.changelabel(curlabel)

        curlabel = label_to_tensor(curlabel)
        return curimg, curlabel, index

    def extract_imgs(self,imgs):
        images = imgs[:,:,(0,1,2)]
        for i in range(self.Images_Norm_params['in_channels']):
            mean = np.squeeze(self.Images_Norm_params['mean'])
            std = np.squeeze(self.Images_Norm_params['std'])
            images[:,:,i] = (images[:,:,i] - mean[i]) / std[i]
        images = image_to_tensor(images)
        return images

    def __getitem__(self,index):
        imgs, labels, idx = self.find_imgs(index)
        return imgs, labels

    def __len__(self):
        return self.num