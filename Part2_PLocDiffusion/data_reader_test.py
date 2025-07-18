from torch.utils.data.dataset import Dataset
from config import *
from tool import *
import cv2
import h5py

class data_reader(Dataset):
    def __init__(self,transform=None):

        self.imgbasedir = TestImgs_basedir
        self.labels = self.read_h5(TestLabels_h5)
        self.transform = transform

    def read_h5(self,datasetsfile):
        h5file = h5py.File(datasetsfile, 'r')
        #data = h5file.get('labels').value
        data = h5file.get('labels')[()]
        return data

    def read_imgs(self, basedir, idx):
        BGimg = cv2.imread(opj(basedir, 'BGYImg' + str(idx) + '.png'))
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
        curlabel = label_to_tensor(curlabel)
        return curimg, curlabel, index

    def __getitem__(self,index):
        imgs, labels, idx = self.find_imgs(index)
        return imgs, labels

    def __len__(self):
        return len(self.labels)