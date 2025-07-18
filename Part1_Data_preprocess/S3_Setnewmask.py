import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import cv2
import numpy as np
import os
import mahotas as mh
import csv
from skimage import filters, morphology

def mkdir(dirName):
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print('---the '+dirName+' is created!---')

def SegImageVoronoi(ImagePath):
    # Use a breakpoint in the code line below to debug your script.
    Single_Flag = 0
    nuclei = cv2.imread(ImagePath)  # 图片读取
    gray = cv2.cvtColor(nuclei, cv2.COLOR_BGR2GRAY)
    # plt.hist(gray)
    # plt.show()
    dnaf = mh.gaussian_filter(gray, 3).astype('uint8')
    thresh = filters.threshold_otsu(dnaf)
    binary = morphology.opening(dnaf > thresh, morphology.disk(5))
    labeled, nr_nuclei = mh.label(binary)
    print(nr_nuclei)
    # plt.imshow(labeled)
    # plt.show()
    seeds, nr_nuclei = mh.labeled.filter_labeled(labeled, remove_bordering=False, min_size=10000)
    # plt.imshow(seeds)
    # plt.show()


    if nr_nuclei <= 1:
        Single_Flag = 1
        return Single_Flag, nuclei, nuclei, nr_nuclei
    else:
        SupplementMask= mh.segmentation.gvoronoi(seeds)
        # plt.imshow(SupplementMask)
        # plt.show()
        return Single_Flag, SupplementMask, nuclei, nr_nuclei


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_dir = 'D:/Program/HPA'
    for tmpFile in os.listdir(train_dir):
        print(tmpFile)
        if tmpFile != 'Lysosomes':
                # 'Plasma membrane_Cytosol' and tmpFile != 'Plasma membrane' \
                # and tmpFile != 'Nucleoplasm_Nucleoli' and tmpFile != 'Nucleoplasm_Mitochondria' \
                # and tmpFile != 'Nucleoli' and tmpFile != 'Mitochondria':
            continue
        else:

            CSV_dir = train_dir + '/' + tmpFile + '/RealSingleCellName.csv'
            FalseCSV_dir = train_dir + '/' + tmpFile + '/FalseSingleCellName.csv'

            f = open(CSV_dir, 'w', encoding='utf-8', newline='')
            csv_writer = csv.writer(f)
            csv_writer.writerow(['Name'])

            Falsef = open(FalseCSV_dir, 'w', encoding='utf-8', newline='')
            Falsecsv_writer = csv.writer(Falsef)
            Falsecsv_writer.writerow(['Name', 'Seeds'])

            oldSingleCell_dir = train_dir + '/' + tmpFile + '/SingleData'
            SupplementMask_dir = train_dir + '/' + tmpFile + '/StepFSupplementMask02_6'
            FalsenewSingleCell_dir = train_dir + '/' + tmpFile + '/StepFFalseSingleData02_6'  # FinalData
            mkdir(SupplementMask_dir)
            mkdir(FalsenewSingleCell_dir)
            for tmpImgMask in os.listdir(oldSingleCell_dir):
                if tmpImgMask.find('blue') != -1:
                    print(tmpImgMask)
                    ImagePath = oldSingleCell_dir + '/' + tmpImgMask
                    Single_Flag, SupplementMask, nuclei, nr_nuclei = SegImageVoronoi(ImagePath)  # Voronoi
                    if Single_Flag == 1:
                        # cv2.imwrite(newSingleCell_dir + '/' + tmpImgMask, nuclei)
                        csv_writer.writerow([tmpImgMask])  # real cell image
                        print('Real Single Cell...')
                    else:
                        cv2.imwrite(FalsenewSingleCell_dir + '/' + tmpImgMask, nuclei)
                        cv2.imwrite(SupplementMask_dir + '/' + tmpImgMask.replace('blue', 'suppleMask'), SupplementMask)
                        Falsecsv_writer.writerow([tmpImgMask, nr_nuclei])  # false cell image
                        print('False Single Cell...')
            f.close()
            Falsef.close()
