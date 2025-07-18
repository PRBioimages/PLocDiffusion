from torch.utils.data.dataset import Dataset
from config.config_ import *
from utils.common_util import *
from datasets.tool import *
import cv2
import pandas as pd

Test_comp = ['CyNp','CyPM','NuMi','NuNuSp']
class data_reader(Dataset):
    def __init__(self,net,Images_Norm_params):

        self.Images_Norm_params = Images_Norm_params
        TestLabels_Classfication_h5 = TestLabelsPrefix+Test_comp[COMB]+TestLabelsSuffix
        if not ope(TestLabels_Classfication_h5):
            raise (FileNotFoundError("The file {} is not found!".format(TestLabels_Classfication_h5)))

        self.net = net
        self.imgbasedir = opj(TestImgs_basedir,Test_comp[COMB])
        labels = self.read_h5_classfication(TestLabels_Classfication_h5)
        self.labels,self.cate_idx = labels,range(len(labels))
        print('len', len(self.labels))

    def read_h5_classfication(self,datasetsfile):
        h5file = h5py.File(datasetsfile, 'r')
        data = h5file['labels'][()]
        return data

    def read_imgs_classfication(self,basedir,idx):
        BGYimg = cv2.imread(opj(basedir, 'BGYImg' + str(idx) + '.png'))
        img = np.zeros([256,256,3],dtype='float32')
        img[:, :, 0] = BGYimg[:, :, 0]
        img[:, :, 1] = BGYimg[:, :, 1]
        img[:, :, 2] = BGYimg[:, :, 2]
        img = img/255
        return img


    def find_imgs(self, index):
        curimg = self.read_imgs_classfication(self.imgbasedir,index)
        curlabel = self.labels[index].astype('float32')
        curimg = self.extract_imgs(curimg)
        curlabel = label_to_tensor(curlabel)
        return curimg, curlabel, index

    def extract_imgs(self,imgs):
        for i in range(self.Images_Norm_params['in_channels']):
            mean = np.squeeze(self.Images_Norm_params['mean'])
            std = np.squeeze(self.Images_Norm_params['std'])
            imgs[:,:,i] = (imgs[:,:,i] - mean[i]) / std[i]
        images = image_to_tensor(imgs)
        return images

    def __getitem__(self,index):
        imgs, labels, idx = self.find_imgs(index)
        return imgs, labels

    def __len__(self):
        return len(self.labels)