import os
from PIL import Image
import pandas as pd
import numpy as np
from skimage import io, filters, segmentation, measure, morphology, color
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter
import random
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def ReadBGRYImg(train_dir,tmpFile,mask_name):
    # print('mask_name:',mask_name)
    samples_dir = train_dir + '/' + tmpFile + '/data'
    blueName = mask_name + '_blue.jpg'
    BlueImgPath = os.path.join(samples_dir, blueName)
    greenName = mask_name + '_green.jpg'
    GreenImgPath = os.path.join(samples_dir, greenName)
    redName = mask_name + '_red.jpg'
    RedImgPath = os.path.join(samples_dir, redName)
    yellowName = mask_name + '_yellow.jpg'
    YellowImgPath = os.path.join(samples_dir, yellowName)

    #
    GreenImg = io.imread(GreenImgPath)[:, :, 1]
    BlueImg = io.imread(BlueImgPath)[:, :, 2]
    YellowImg = np.asarray(Image.open(YellowImgPath).convert('L'))
    RedImg = io.imread(RedImgPath)[:, :, 0]
    return GreenImg, BlueImg, YellowImg, RedImg

def mkdir(dirName):
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print('---the '+dirName+' is created!---')
    else:
        print('---The dir is there!---')

if __name__ == "__main__":
    initialthre = 10000
    NumCell = 100
    train_dir = 'D:/Program/HPA'
    filenum = 0
    for tmpFile in os.listdir(train_dir):
        print(tmpFile)
        filenum = filenum + 1
        if filenum <= 0:
            continue
        else:
            mask_dir = train_dir + '/' + tmpFile + '/mask'
            SegMask_dir = train_dir + '/' + tmpFile + '/MaskSeg'
            SegData_dir = train_dir + '/' + tmpFile + '/SingleData/'
            mkdir(SegMask_dir)
            mkdir(SegData_dir)

            for tmpImgMask in os.listdir(mask_dir):
                print(tmpImgMask)
                if tmpImgMask.find('fp') !=-1:
                    print('The Image did not download correctly!')
                    continue
                tmpmask_name = tmpImgMask.split("_segmentation")
                mask_name = tmpmask_name[0]
                Dulicate_Flag = 0
                for tmpImgSeg in os.listdir(SegMask_dir):
                    tmpSeg_name = tmpImgSeg.split("_segmentation")
                    Seg_name = tmpSeg_name[0]
                    if mask_name == Seg_name:
                        Dulicate_Flag = 1

                if Dulicate_Flag == 1:
                    print('Dulicate...')
                else:
                # if 1:
                    # m_path = os.path.join(mask_dir, tmpImgMask)
                    img = cv2.imread(mask_dir + '/' + tmpImgMask)  # 图片读取
                    # img = io.imread(m_path)
                    # img = np.array(img)
                    # plt.imshow(img)
                    # plt.show()
                    # cv2.imshow('picture', img)
                    # cv2.waitKey(0)
                    pictue_size = img.shape
                    picture_height = pictue_size[0]
                    picture_width = pictue_size[1]
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # ret, BinaryImg = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
                    retBinary, BinaryImg = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
                    # BinaryImg = mahotas.thresholding.ostu(gray)
                    # cv2.imshow('picture', BinaryImg)
                    # cv2.waitKey(0)

                    # gaussian filter : filt some little pixel patches
                    GaussianFiltImg = gaussian_filter(BinaryImg, 4)
                    # plt.imshow(GaussianFiltImg)
                    # plt.show()
                    # GaussianBinaryImg = mahotas.thresholding.ostu(GaussianFiltImg)
                    retGaussianBinary, GaussianBinaryImg = cv2.threshold(GaussianFiltImg, 0, 255, cv2.THRESH_OTSU)
                    # plt.imshow(GaussianBinaryImg)
                    # plt.show()
                    cleared = GaussianBinaryImg.copy()  # copy

                    segmentation.clear_border(cleared)  # clear up the noise

                    label_image = measure.label(cleared)  # label the region
                    # print(len(label_image))
                    # print(label_image.shape)
                    # if len(label_image) > NumCell:
                        # continue

                    StoreNum = 0
                    sumArea = 0
                    if len(measure.regionprops(label_image)) > NumCell:
                        print('Num%d'%len(measure.regionprops(label_image)),' is too large!')
                        continue
                    # print(len(measure.regionprops(label_image)))
                    for region1 in measure.regionprops(label_image):
                        sumArea = region1.area + sumArea
                    MeanCellArea = sumArea/len(measure.regionprops(label_image))
                    for region in measure.regionprops(label_image):  # 循环得到每一个连通区域属性集
                        # 忽略小区域
                        if sumArea * 0.0026 > initialthre:
                            thre = sumArea
                        else:
                            thre = initialthre
                        # print(thre)
                        if region.area < thre:
                            print('Too Small...')
                            continue
                        else:
                            # if SingleCellArea[i] >  lower_quartile - 1.5 * (upper_quartile-lower_quartile):
                            bm = np.zeros(shape=label_image.shape)
                            SegMaskImg = np.zeros(shape=label_image.shape)
                            NumColEndBoundary = 0
                            NumColBeginBoundary = 0
                            NumRowEndBoundary = 0
                            NumRowBeginBoundary = 0
                            Boundary_Flag = 1
                            for coord in region.coords:
                                if region.area < MeanCellArea:
                                    if coord[0] >= picture_width - 5:
                                        NumColEndBoundary = NumColEndBoundary + 1
                                        # print('(a,b):(', coord[1], ',', coord[0], ')')
                                    elif coord[0] <= 4:
                                        NumColBeginBoundary = NumColBeginBoundary + 1
                                        # print('(a,b):(', coord[1], ',', coord[0], ')')
                                    elif coord[1] >= picture_height - 5:
                                        NumRowEndBoundary = NumRowEndBoundary + 1
                                        # print('(a,b):(', coord[1], ',', coord[0], ')')
                                    elif coord[1] <= 4:
                                        NumRowBeginBoundary = NumRowBeginBoundary + 1
                                        # print('(a,b):(', coord[1], ',', coord[0], ')')
                                    if NumColEndBoundary >= 10 or NumColBeginBoundary >= 10 \
                                            or NumRowEndBoundary >= 10 or NumRowBeginBoundary >= 10:
                                        # print('Boundary ERROR')
                                        Boundary_Flag = 0
                                if Boundary_Flag == 1:
                                    bm[coord[0], coord[1]] = 1

                            # plt.imshow(bm)
                            # plt.show()
                            if Boundary_Flag == 1:
                                GreenImg, BlueImg, YellowImg, RedImg = ReadBGRYImg(train_dir, tmpFile, mask_name)
                                GreenImg = bm * GreenImg
                                BlueImg = bm * BlueImg
                                YellowImg = bm * YellowImg
                                RedImg = bm * RedImg
                                # print(max(bm[:]))
                                minr, minc, maxr, maxc = region.bbox
                                g_patch = GreenImg[minr:maxr, minc:maxc]
                                b_patch = BlueImg[minr:maxr, minc:maxc]
                                y_patch = YellowImg[minr:maxr, minc:maxc]
                                r_patch = RedImg[minr:maxr, minc:maxc]
                                bm = 255*bm[minr:maxr, minc:maxc]

                                StoreNum = StoreNum + 1
                                # print('Mask Num:', StoreNum)
                                Maskname = tmpImgMask.split(".")
                                Maskstoredir = SegMask_dir + '/'+Maskname[0] + str(StoreNum)
                                BGRYstoredir = SegData_dir + Maskname[0] + str(StoreNum)
                                # print(storedir)
                                maskdir = Maskstoredir + '.png'
                                bluedir = BGRYstoredir + '_blue.jpg'
                                greendir = BGRYstoredir + '_green.jpg'
                                reddir = BGRYstoredir + '_red.jpg'
                                yellowdir = BGRYstoredir + '_yellow.jpg'
                                cv2.imwrite(maskdir, bm)
                                cv2.imwrite(bluedir, b_patch)
                                cv2.imwrite(greendir, g_patch)
                                cv2.imwrite(reddir, r_patch)
                                cv2.imwrite(yellowdir, y_patch)