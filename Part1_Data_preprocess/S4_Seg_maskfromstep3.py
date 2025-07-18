import os
from PIL import Image
import pandas as pd
import numpy as np
from skimage import io, filters, segmentation, measure, morphology, color
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter
import random
from PIL import Image, ImageOps
import math
def mkdir(dirName):
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print('---the '+dirName+' is created!---')
    else:
        print('---The dir is there!---')

def expand_img(grayBlueImg, boundary):
    w = grayBlueImg.size[0]
    h = grayBlueImg.size[1]
    pr = 0
    pc = 0
    cr = 0
    cc = 0
    r = int(math.fabs(boundary - w) / 2 + 0.5)
    c = int(math.fabs(boundary - h) / 2 + 0.5)
    if w <= boundary:
        pr = r
    else:
        print('Width over 1024')
        cr = r
    if h <= boundary:
        pc = c
    else:
        cc = c
        print('Height over 1024')

    # print(pr, pc, pr, pc)
    pad_b_img = ImageOps.expand(grayBlueImg, (pr, pc, pr, pc))
    resize_b_img = pad_b_img.resize((boundary, boundary), Image.ANTIALIAS)
    resize_b_img = np.array(resize_b_img).astype(np.uint8)
    return resize_b_img

def MaskCrop(cropImagePath, SupplementMask):
    cropImage = cv2.imread(cropImagePath)
    pictue_size = cropImage.shape
    picture_height = pictue_size[0]
    picture_width = pictue_size[1]
    r = int(math.fabs(2048 - picture_width) / 2 + 0.5)
    c = int(math.fabs(2048 - picture_height) / 2 + 0.5)
    minc = r
    maxc = r + picture_width
    minr = c
    maxr = c + picture_height
    SupplementMask = SupplementMask[minr:maxr, minc:maxc]
    return SupplementMask

def GetSingleMask(SupplementMask, maxV, minV, needV):
    # plt.imshow(SupplementMask)
    # plt.show()
    if needV != minV and needV != maxV:
        retMaskBinaryImg1, MaskBinaryImg1 = cv2.threshold(SupplementMask, needV-1, 255, cv2.THRESH_BINARY)
        # plt.imshow(MaskBinaryImg1)
        # plt.show()
        retMaskBinaryImg2, MaskBinaryImg2 = cv2.threshold(SupplementMask, needV, 255, cv2.THRESH_BINARY)
        # plt.imshow(MaskBinaryImg2)
        # plt.show()
        MaskBinaryImg = MaskBinaryImg1 - MaskBinaryImg2
    elif needV == minV:
        retMaskBinaryImg, MaskBinaryImg = cv2.threshold(SupplementMask, needV, 255, cv2.THRESH_BINARY)
        MaskBinaryImg = 255 - MaskBinaryImg
    elif needV == maxV:
        retMaskBinaryImg, MaskBinaryImg = cv2.threshold(SupplementMask, needV-1, 255, cv2.THRESH_BINARY)
        MaskBinaryImg = MaskBinaryImg
    # plt.imshow(MaskBinaryImg)
    # plt.show()
    return MaskBinaryImg


def ReadBGRYImg(train_dir,tmpFile,mask_name):
    # print('mask_name:',mask_name)
    samples_dir = train_dir + '/' + tmpFile + '/SingleData'
    mask_dir = train_dir + '/' + tmpFile + '/' + 'MaskSeg'
    blueName = mask_name + '_blue.jpg'
    BlueImgPath = os.path.join(samples_dir, blueName)
    greenName = mask_name + '_green.jpg'
    GreenImgPath = os.path.join(samples_dir, greenName)
    redName = mask_name + '_red.jpg'
    RedImgPath = os.path.join(samples_dir, redName)
    yellowName = mask_name + '_yellow.jpg'
    YellowImgPath = os.path.join(samples_dir, yellowName)
    maskcropName = mask_name + '.png'
    maskcropPath = os.path.join(mask_dir, maskcropName)
    #
    GreenImg = Image.open(GreenImgPath)
    BlueImg = Image.open(BlueImgPath)
    YellowImg = Image.open(YellowImgPath)
    RedImg = Image.open(RedImgPath)
    maskImg = Image.open(maskcropPath)

    # GreenImg = expand_img(GreenImg, 2048)
    # BlueImg = expand_img(BlueImg, 2048)
    # YellowImg = expand_img(YellowImg, 2048)
    # RedImg = expand_img(RedImg, 2048)
    # maskImg = expand_img(maskImg, 2048)

    # GreenImg = cv2.imread(GreenImgPath)
    # BlueImg = cv2.imread(BlueImgPath)
    # YellowImg = cv2.imread(YellowImgPath)
    # RedImg = cv2.imread(RedImgPath)
    # maskImg = cv2.imread(maskcropPath)
    #
    # GreenImg = cv2.cvtColor(GreenImg, cv2.COLOR_BGR2GRAY)
    # BlueImg = cv2.cvtColor(BlueImg, cv2.COLOR_BGR2GRAY)
    # YellowImg = cv2.cvtColor(YellowImg, cv2.COLOR_BGR2GRAY)
    # RedImg = cv2.cvtColor(RedImg, cv2.COLOR_BGR2GRAY)
    # maskImg = cv2.cvtColor(maskImg, cv2.COLOR_BGR2GRAY)

    return GreenImg, BlueImg, YellowImg, RedImg, maskImg


def CropImage(MaskBinaryImg, train_dir, tmpFile, mask_name):
    thresh = filters.threshold_otsu(MaskBinaryImg)
    bw = morphology.opening(MaskBinaryImg > thresh, morphology.disk(3))
    cleared = bw.copy()  # copy
    segmentation.clear_border(cleared)  # clear up the noise
    # plt.imshow(cleared)
    # plt.show()
    label_image = measure.label(cleared)  # label the region
    for region in measure.regionprops(label_image):
        bm = np.zeros(shape=label_image.shape)
        for coord in region.coords:
            bm[coord[0], coord[1]] = 1
        GreenImg, BlueImg, YellowImg, RedImg, maskImg = ReadBGRYImg(train_dir, tmpFile, mask_name)
        GreenImg = bm * GreenImg
        BlueImg = bm * BlueImg
        YellowImg = bm * YellowImg
        RedImg = bm * RedImg
        maskImg = bm * maskImg
        # print(max(bm[:]))

        minr, minc, maxr, maxc = region.bbox
        g_patch = GreenImg[minr:maxr, minc:maxc]
        b_patch = BlueImg[minr:maxr, minc:maxc]
        y_patch = YellowImg[minr:maxr, minc:maxc]
        r_patch = RedImg[minr:maxr, minc:maxc]
        maskImg = maskImg[minr:maxr, minc:maxc]
        bm = 255 * bm[minr:maxr, minc:maxc]
        return g_patch, b_patch, y_patch, r_patch, maskImg

train_dir = 'D:/Program/HPA'

for tmpFile in os.listdir(train_dir):
    print(tmpFile)
    if tmpFile != 'Lysosomes':
        continue
    else:
        SuppleMask_dir = train_dir + '/' + tmpFile + '/' + 'StepFSupplementMask02_6'
        cropImage_dir = train_dir + '/' + tmpFile + '/' + 'StepFFalseSingleData02_6'
        newstoredata_dir = train_dir + '/' + tmpFile + '/' + 'StepGNewSuppleSingleData_7'
        newstoremask_dir = train_dir + '/' + tmpFile + '/' + 'StepGNewSuppleSingleMask_7'
        mkdir(newstoredata_dir)
        mkdir(newstoremask_dir)
        FileCSV_dir = train_dir + '/' + tmpFile
        FileCSVPath = FileCSV_dir + '/' + 'FalseSingleCellName.csv'
        print(FileCSVPath)
        FalseSingleCell = pd.read_csv(FileCSVPath)
        FalseSingCellname = FalseSingleCell["Name"]  #获取一列，用一维数据
        FalseSingCellname = np.array(FalseSingCellname)
        FalseSingCellseeds = FalseSingleCell["Seeds"]  #获取一列，用一维数据
        FalseSingCellseeds = np.array(FalseSingCellseeds)


        for tmpImgMask in os.listdir(SuppleMask_dir):
            print(tmpImgMask)
            cropImageName = tmpImgMask.replace('_suppleMask', '_blue')
            mask_name = tmpImgMask.split('_suppleMask')[0]
            cropImagePath = cropImage_dir + '/' + cropImageName
            SupplementMaskPath = SuppleMask_dir + '/' + tmpImgMask
            SupplementMask = cv2.imread(SupplementMaskPath)
            SupplementMask = cv2.cvtColor(SupplementMask, cv2.COLOR_BGR2GRAY)
            for Num in range(len(FalseSingCellname)):
                tmpName = FalseSingCellname[Num]
                if tmpName == cropImageName:
                    maxV = FalseSingCellseeds[Num]  # cells number
                    break
            minV = 1
            print(minV, maxV)
            # SupplementMask = MaskCrop(cropImagePath, SupplementMask)  # crop mask as large as image in 'Maskchosen' file
            # plt.imshow(SupplementMask)
            # plt.show()

            for needV in range(minV, maxV+1):
                print(needV)
                MaskBinaryImg = GetSingleMask(SupplementMask, maxV, minV, needV)  # get single Mask
                g_patch, b_patch, y_patch, r_patch, bm = CropImage(MaskBinaryImg, train_dir, tmpFile, mask_name)  # seg cell
                # store
                Maskname = tmpImgMask.split(".")
                Maskstoredir = newstoremask_dir + '/' + mask_name + '_' + str(needV)
                BGRYstoredir = newstoredata_dir + '/' + mask_name + '_supple' + str(needV)
                # print(storedir)
                maskdir = Maskstoredir + '_suppleMask.png'
                bluedir = BGRYstoredir + '_blue.jpg'
                greendir = BGRYstoredir + '_green.jpg'
                reddir = BGRYstoredir + '_red.jpg'
                yellowdir = BGRYstoredir + '_yellow.jpg'
                cv2.imwrite(maskdir, bm)
                cv2.imwrite(bluedir, b_patch)
                cv2.imwrite(greendir, g_patch)
                cv2.imwrite(reddir, r_patch)
                cv2.imwrite(yellowdir, y_patch)
