from torch.utils.data.dataset import Dataset
from config import *
from tool import *
import cv2
import pickle
import pandas as pd
import h5py
osp = os.path
ope = os.path.exists
opj = os.path.join
class data_reader(Dataset):
    def __init__(self, is_Train=True, transform=None):

        if not ope(TrainLabels_h5):
            raise (FileNotFoundError("The file {} is not found!".format(TrainLabels_h5)))
        if not ope(ValLabels_h5):
            raise (FileNotFoundError("The file {} is not found!".format(ValLabels_h5)))
        if not ope(TrainImgs_basedir):
            raise (FileNotFoundError("The file {} is not found!".format(TrainImgs_basedir)))
        if not ope(ValImgs_basedir):
            raise (FileNotFoundError("The file {} is not found!".format(ValImgs_basedir)))

        self.is_Train = is_Train
        self.transform = transform

        if is_Train:
            self.imgbasedir = TrainImgs_basedir
            self.labels = self.read_h5(TrainLabels_h5)
        else:
            self.imgbasedir = ValImgs_basedir
            self.labels = self.read_h5(ValLabels_h5)

    def read_h5(self,datasetsfile):
        h5file = h5py.File(datasetsfile, 'r')
        data = h5file.get('labels')[()]
        return data

    def changelabel(self, label):
        partial_label = [[0.75, 0.25], [0.5, 0.5], [0.25, 0.75]]
        tmplabel = label > 0
        if (tmplabel == [1, 1]).all():
            idx = np.random.randint(0, 3)
            finalable = partial_label[idx]
        else:
            finalable = label
        return np.array(finalable)


    def read_imgs(self,basedir,idx):
        BGimg = cv2.imread(opj(basedir,'BGYImg'+str(idx)+'.png'))
        img = np.zeros([256, 256, 3], dtype='float32')
        img[:, :, 0] = BGimg[:, :, 0]
        img[:, :, 1] = BGimg[:, :, 1]
        img[:, :, 2] = BGimg[:, :, 2]
        if self.transform is not None:
            img = self.transform(img)
        img = img / 127.5 - 1
        return img

    def find_imgs(self, index):
        curimg = self.read_imgs(self.imgbasedir,index)
        curlabel = self.labels[index].astype('float32')
        curimg = image_to_tensor(curimg)
        curlabel = self.changelabel(curlabel)
        curlabel = label_to_tensor(curlabel)
        return curimg, curlabel, index

    def __getitem__(self,index):
        imgs, labels, idx = self.find_imgs(index)
        return imgs, labels

    def __len__(self):
        return len(self.labels)