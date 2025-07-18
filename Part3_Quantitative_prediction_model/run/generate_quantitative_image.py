import h5py
import numpy as np
import cv2
import os
import h5py
from utils.augment_util_gan import train_gan_augment

baselabel = {
    0: [1, 0, 0, 0, 0],
    1: [0, 1, 0, 0, 0],
    2: [0, 0, 1, 0, 0],
    3: [0, 0, 0, 1, 0],
    4: [0, 0, 0, 0, 1],
    5: [1, 1, 0, 0, 0],
    6: [1, 0, 0, 0, 1],
    7: [0, 1, 1, 0, 0],
    8: [0, 1, 0, 1, 0]
}


def read_h5_classfication(datasetsfile):
    h5file = h5py.File(datasetsfile, 'r')
    data = h5file['labels'][:]
    return data

def read_imgs_classfication(basedir, idx):
    BGYimg = cv2.imread(os.path.join(basedir, 'BGYImg' + str(idx) + '.png'))
    img = np.zeros([256, 256, 3], dtype='float32')
    img[:, :, 0] = BGYimg[:, :, 0]
    img[:, :, 1] = BGYimg[:, :, 1]
    img[:, :, 2] = BGYimg[:, :, 2]
    img = img / 127.5 - 1
    return img

def label_5to2(label):
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
    for i in baselabel.keys():
        if (baselabel[i] == binlabel).all():
            comb = label_rev_anno[i]
            break
    return comb

def changelabel(label):
    idx = np.where(label == 1)
    if len(label[idx]) == 2:
        comb = label_5to2(label)
        if comb == 'None':
            raise ValueError('comb can not be None!')
        idx = np.array(idx).squeeze()
        chantmplabel = [0.5, 0.5]
        for n in range(2):
            label[idx[n]] = chantmplabel[n]
    return label

def save_to_h5(output_file, curlabels):
    with h5py.File(output_file, 'w') as h5file:
        h5file.create_dataset('labels', data=curlabels)
        print(f"Labels saved to {output_file}")


def main(datasetsfile, basedir, output_base_dir):
    labels = read_h5_classfication(datasetsfile)
    for p_ratio in range(15, 100, 15):
        for repeat in range(0, 5):
            curlabels = []

            p0 = p_ratio / 100.0
            remaining = 1 - p0
            p = [remaining, p0 / 3.0, p0 / 3.0, p0 / 3.0]

            output_dir = os.path.join(output_base_dir, f"p_{p_ratio}%_run{repeat + 1}")
            output_h5 = os.path.join(output_base_dir, f"Trainlabels_p{p_ratio}_run{repeat + 1}.h5")

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            i = 0
            for idx, label in enumerate(labels):
                original_label = labels[idx].astype('float32')
                img = read_imgs_classfication(basedir, idx)
                idx_1 = np.where(original_label == 1)
                comb = label_5to2(label)
                if len(original_label[idx_1]) == 2:
                    img_a, cur_label, flag = train_gan_augment(img, original_label, p)
                    if flag == 1:
                        output_path = os.path.join(output_dir, 'BGYImg' + str(i) + '.png')
                        img_to_save = np.clip((img_a + 1) * 127.5, 0, 255).astype(np.uint8)
                        cv2.imwrite(output_path, img_to_save)
                        curlabels.append(cur_label)
                        i = i + 1
                else:
                    continue
            curlabels = np.array(curlabels)
            save_to_h5(output_h5, curlabels)

        # 保存 curlabels 到 h5 文件
    curlabels = np.array(curlabels)
    save_to_h5(output_h5, curlabels)


#     # 示例调用主函数
TrainImgs_basedir = '../TrainImgs'
TrainLabels_Classfication_h5 = '../AllRCateTrainLabelsU2RforClassfication_.h5'
output_dir = "../TrainImgs_allp"

main(TrainLabels_Classfication_h5, TrainImgs_basedir, output_dir)



