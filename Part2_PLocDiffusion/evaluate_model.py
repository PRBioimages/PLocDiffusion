import numpy as np
import csv
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from PIL import Image
import lpips
from fid_score.fid_score import FidScore
from .prdc import compute_prdc
from .ms_ssim import ms_ssim
from torchvision import models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import torch
import math
import os
import cv2

img_to_tensor = transforms.ToTensor()
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def create_csv(path, csv_head):
    with open(path, 'w') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(csv_head)


def write_csv(data_row, path):
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)
        # print("write over")


def metricevalue(fakeimgLabel, realimgLabel, labels,batch_size,path, thre=0.5):
    measureindx = []
    measurenum = 0
    TP = 0
    FN = 0
    FP = 0
    TN = 0

    for i in range(batch_size):
        tmplabel = labels[i]
        measurenum = measurenum + 2
        measureindx.append(i)
        tmpfakeimgLabel = fakeimgLabel[i]
        tmprealimgLabel = realimgLabel[i]
        tmpfakeimgFrac = tmpfakeimgLabel
        tmprealimgFrac = tmprealimgLabel
        if tmprealimgFrac > thre:
            TP = TP + 1
        elif tmprealimgFrac <= thre:
            FN = FN + 1
        if tmpfakeimgFrac > thre:
            FP = FP + 1
        elif tmpfakeimgFrac <= thre:
            TN = TN + 1
    acc = (TP + TN) / measurenum
    dice = 2 * TP / (FP + 2 * TP + FN)
    data_row = [str(TP), str(FN), str(FP), str(TN), str(measurenum)]
    write_csv(data_row, path)
    return TP, FN, FP, TN, measurenum, acc, dice


def Acc_Dice(TP=0, FN=0, FP=0, TN=0, measurenum=0):
    acc = (TP + TN) / measurenum
    dice = 2 * TP / (FP + 2 * TP + FN)
    return acc, dice


def labelAcc(fakeimgLabel, labels, thr,batch_size):
    TP = 0
    for i in range(batch_size):
        tmplabel = labels[i]
        tmpfakelabel = fakeimgLabel[i]
        dis = ((tmplabel[0] - tmpfakelabel[0]) ** 2 + (tmplabel[1] - tmpfakelabel[1]) ** 2) / 2
        if dis <= thr:
            TP = TP + 1
    acc = TP / batch_size
    return TP, acc

def multi_class_acc(preds, targs, th=10e-2):
    pred = preds
    targ = targs
    diff_sum = []
    for i in range(len(pred)):
        flag = 0
        for n in range(2):
            _pred = pred[i,n]
            _targ = targ[i,n]
            if _targ == 0 :
                if _pred <= 0.05:
                    flag +=1
            if _targ == 0.25:
                if _pred > 0.05 and _pred <= 0.375:
                    flag += 1
            if _targ == 0.5:
                if _pred > 0.375 and _pred <= 0.625:
                    flag += 1
            if _targ == 0.75:
                if _pred > 0.625 and _pred <= 0.95:
                    flag += 1
            if _targ == 1:
                if _pred > 0.95 and _pred <= 1:
                    flag += 1
        if flag == 2:
            diff_sum.append(1)
        else:
            diff_sum.append(0)
    return np.array(diff_sum).mean(),np.array(diff_sum)

def cal_ssim(im1, im2,label,singley=False):
    value = []
    if singley:
        for i in range(len(label)):
            tmplabel = label[i]
            if tmplabel[0] == 1 or tmplabel[0] == 0:
                imgreal = im1[i, :, :, 1]
                imgfake = im2[i, :, :, 0]
                value.append(ssim(imgreal, imgfake,data_range=255))
        return value, np.mean(value)
    else:
        for i in range(len(label)):
            imgreal = im1[i, :, :, 1]
            imgfake = im2[i, :, :, 0]
            value.append(ssim(imgreal, imgfake,data_range=255))
        return value, np.mean(value)


def cal_mse(im1, im2,batch_size,label,singley=False):
    value = []
    if singley:
        for i in range(batch_size):
            tmplabel = label[i]
            if tmplabel[0] == 1 or tmplabel[0] == 0:
                imgreal = im1[i, :, :, 1]
                imgfake = im2[i, :, :, 0]
                value.append(mse(imgreal, imgfake))
        return value, np.mean(value)
    else:
        for i in range(batch_size):
            imgreal = im1[i, :, :, 1]
            imgfake = im2[i, :, :, 0]
            value.append(mse(imgreal, imgfake))
        return value, np.mean(value)


def evaluate_lpips(img1,img2):

    loss_fn_vgg = lpips.LPIPS(net='vgg', verbose=False).cuda()

    img1_tens = TF.to_tensor(img1).unsqueeze(0) * 2 - 1
    img2_tens = TF.to_tensor(img2).unsqueeze(0) * 2 - 1

    img1_tens = img1_tens.to('cuda')
    img2_tens = img2_tens.to('cuda')

    lpipsmetric = loss_fn_vgg.forward(img1_tens, img2_tens)
    print(np.squeeze(lpipsmetric.cpu().detach().numpy()))
    return np.squeeze(lpipsmetric.cpu().detach().numpy())

def evaluate_fid(paths,batch_size):
    device = torch.device('cuda:0')
    fid = FidScore(paths, device, batch_size)
    score = fid.calculate_fid_score()
    # print(score)
    return score

def psnr(tf_img1, tf_img2, max_val):
    MSEimg = mse(tf_img1, tf_img2)
    psnr = 20 * math.log10(max_val / math.sqrt(MSEimg))
    return psnr

def cal_psnr(im1, im2,label,singley=False):
    value = []
    if singley:
        for i in range(len(label)):
            tmplabel = label[i]
            if tmplabel[0] == 1 or tmplabel[0] == 0:
                imgreal = im1[i, :, :, 1]
                imgfake = im2[i, :, :, 0]
                value.append(psnr(imgreal, imgfake, max_val=255))
        return value, np.mean(value)
    else:
        for i in range(len(label)):
            imgreal = im1[i, :, :, 1]
            imgfake = im2[i, :, :, 0]
            value.append(psnr(imgreal, imgfake,max_val=255))
        return value, np.mean(value)

def extract_features(img):
    net = models.vgg16(pretrained=True).to('cuda')
    net.eval()
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    img3d = img.reshape(1, 224, 224).repeat(3, axis=0)
    input = img_to_tensor(img3d)
    input = input.reshape(1, 3, 224, 224).float().cuda()
    ifeatures = net.features(input)
    ifeatures = net.avgpool(ifeatures)
    ifeatures = torch.flatten(ifeatures, 1)
    features = net.classifier[:4](ifeatures).cpu().detach().numpy()
    return features

def cal_coverage_density(real_features,fake_features,nearest_k):
    metrics = compute_prdc(real_features=real_features,
                           fake_features=fake_features,
                           nearest_k=nearest_k)
    coverage = metrics['coverage']
    density = metrics['density']
    return coverage,density

def cal_msssim(im1, im2,batch_size,label,data_range=1,singley=False):
    value = []
    if singley:
        for i in range(batch_size):
            tmplabel = label[i]
            if tmplabel[0] == 1 or tmplabel[0] == 0:
                imgreal = torch.cuda.FloatTensor(im1[i, :, :, 1]).view(1,1, 256, 256)
                imgfake = torch.cuda.FloatTensor(im2[i, :, :, 0]).view(1,1, 256, 256)
                value.append(ms_ssim(imgreal, imgfake, data_range, size_average=False).cpu().detach().numpy())
        return value, np.mean(value)
    else:
        for i in range(batch_size):
            imgreal = torch.cuda.FloatTensor(im1[i, :, :, 1]).view(1,1, 256, 256)
            imgfake = torch.cuda.FloatTensor(im2[i, :, :, 0]).view(1,1, 256, 256)
            value.append(ms_ssim(imgreal, imgfake, data_range, size_average=False).cpu().detach().numpy())
        return value, np.mean(value)

def cal_psnr_ssim(opts,ep,sum_ssim,sum_psnr,im1, im2,label,singley=False):
    num = 0
    batch_size = opts.batch_size
    ssimvalueB = []
    psnrvalueB = []
    if len(sum_psnr) == 0:
        ssim_value = []
        psnr_value = []
    else:
        ssim_value = sum_ssim
        psnr_value = sum_psnr
    if singley:

        for i in range(batch_size):
            tmplabel = label[i]
            if tmplabel[0] == 1 or tmplabel[0] == 0:
                num = num +1
                imgreal = im1[i, :, :, 1]
                imgfake = im2[i, :, :, 0]
                tmppsnr = psnr(imgreal, imgfake, max_val=1)
                tmpssim = ssim(imgreal, imgfake,data_range=1)
                if len(ssim_value) == 0:
                    ssim_value = tmpssim
                    psnr_value = tmppsnr
                else:
                    psnr_value = np.append(psnr_value, tmppsnr, axis=0)
                    ssim_value = np.append(ssim_value, tmpssim, axis=0)
                psnrvalueB.append(tmppsnr)
                ssimvalueB.append(tmpssim)
                iter = ep*opts.batch_size+num
                # store_display(opts, iter, imgreal, imgfake, ssim=np.mean(ssim_value), psnr=np.mean(psnr_value))
        return psnrvalueB,ssimvalueB
    else:
        for i in range(batch_size):
            num = num + 1
            imgreal = im1[i, :, :, 1]
            imgfake = im2[i, :, :, 0]
            tmppsnr = psnr(imgreal, imgfake, max_val=1)
            tmpssim = ssim(imgreal, imgfake,data_range=1)
            if np.size(ssim_value) == 0:
                ssim_value = tmpssim
                psnr_value = tmppsnr
            else:
                psnr_value = np.append(psnr_value,tmppsnr)
                ssim_value = np.append(ssim_value, tmpssim)
            psnrvalueB.append(tmppsnr)
            ssimvalueB.append(tmpssim)
            iter = ep * opts.batch_size + num
            real = np.zeros((opts.height , opts.height, 3))
            real[:,:,1] = imgreal
            fake = np.zeros((opts.height, opts.height, 3))
            fake[:, :, 1] = imgfake
            # store_display(opts, iter, real, fake, ssim=np.mean(ssim_value), psnr=np.mean(psnr_value))
        return psnrvalueB, ssimvalueB


def calc_lpips(opts,data,model,singley=False):
    samples = []
    labels = []
    lpipssum = []
    iter = 0
    if singley:
        for k in range(80):
            torch.cuda.empty_cache()
            print('epoch:%d' % k)
            x, y = data.next_test_batch(opts.batch_size)
            singlelabel = 0
            singleyidx = []
            for ilabel in range(opts.batch_size):
                tmplabel = y[ilabel]
                if tmplabel[0] == 1 or tmplabel[0] == 0:
                    singlelabel = singlelabel+1
                    if len(singleyidx) ==0:
                        singleyidx = ilabel
                    else:
                        singleyidx = np.append(singleyidx,ilabel)
            print('Num Single:%d'%singlelabel,'--Single Idx:',singleyidx)

            for i in range(10):
                output_test, _ = model.test_forward(x, y)
                if i == 0:
                    samples = output_test[singleyidx]
                else:
                    samples = np.append(samples, output_test[singleyidx], axis=0)
                labels.extend(y[singleyidx])
            for imgnum in range(singlelabel):
                calcdata = samples[np.array(range(10)) * singlelabel + imgnum, :, :]
                for n in range(10):
                    for m in range(n + 1, 10):
                        iter = iter + 1
                        tmplpips = evaluate_lpips(calcdata[n, :, :], calcdata[m, :, :])
                        lpipssum.append(tmplpips)
                        img1 = np.zeros((opts.height, opts.height, 3))
                        img1[:, :, 1] = calcdata[n, :, :]
                        img2 = np.zeros((opts.height, opts.height, 3))
                        img2[:, :, 1] = calcdata[m, :, :]
                        # store_display(opts, iter, img1, img2, lpips=np.mean(lpipssum),enDiv=True)
            print('=========================================LPIPS %04f' % np.mean(lpipssum))
        return np.mean(lpipssum)

    else:
        for iter, iter_data in enumerate(data, 0):
            torch.cuda.empty_cache()
            print('epoch:%d' % iter)
            if iter > 79:
                break
            x, y = iter_data
            for i in range(10):
                # output_test,_ = model.test_forward_(x, y)
                output_test = np.zeros((len(x), 256, 256, 1),dtype='float32')
                for n in range(len(x)):
                    output_test[n, :, :, 0] = cv2.imread(('../Genimage/Sum_U2La_/' +
                                                                   'epoch_%d_batch_%d_#img_%d.png') % (iter, n, i))[:,:, 1]
                if i == 0:
                    samples = output_test
                else:
                    samples = np.append(samples, output_test, axis=0)
                labels.extend(y)
            for imgnum in range(opts.batch_size):
                calcdata = samples[np.array(range(10))*opts.batch_size+imgnum,:,:]
                ilpips = 0
                for n in range(10):
                    for m in range(n+1,10):
                        tmplpips = evaluate_lpips(calcdata[n,:,:],calcdata[m,:,:])
                        if iter == 0:
                            if not os.path.exists(opts.sampledir):
                                os.mkdir(opts.sampledir)
                            create_csv(opts.test_Metric,csv_head=["Iter","Epoch","Idx", "LPIPS"])
                        data_row = [str(iter),str(iter),str(ilpips), str(tmplpips)]
                        write_csv(data_row, opts.test_Metric)
                        iter = iter + 1
                        ilpips = ilpips + 1
                        lpipssum.append(tmplpips)
                        img1 = np.zeros((opts.height, opts.height, 3))
                        img1[:, :, 1] = calcdata[n, :, :,0]
                        img2 = np.zeros((opts.height, opts.height, 3))
                        img2[:, :, 1] = calcdata[m, :, :,0]
                        # store_display(opts, iter, img1, img2, lpips=np.mean(lpipssum), enDiv=True)

            print('=========================================LPIPS %04f'%np.mean(lpipssum))
        return np.mean(lpipssum)


def diversity_genimg(opts,data,suffix=None):
    sum_store_path = os.path.join('../../Fidrealimg/'+opts.comb+'_'+suffix,'Sum')
    if not os.path.exists(sum_store_path):
        os.makedirs(sum_store_path)
    i = 0
    for iter, iter_data in enumerate(data, 0):
        if i >999:
            break
        inputs, labels = iter_data
        inputs = inputs.permute(0, 2, 3, 1).cpu().detach().numpy()
        for n in range(len(labels)):
            res2 = np.zeros((opts.height, opts.height,3))
            res2[:, :, 1] = inputs[n, :, :, 1]
            cv2.imwrite(os.path.join(sum_store_path, 'Fidreal_epoch%d_batch%d_img.png') % (iter, n),
                        (res2 * 255).astype(np.uint8))
            i = i + 1

def calc_fid(opts,result_dir,suffix=None,ensum=False):

    gen_batch = 2 * opts.batch_size
    if ensum:
        ireal = os.path.join('../../Fidrealimg/'+opts.comb+'_'+suffix,'Sum')
        ifake = os.path.join(result_dir, 'Sum_'+opts.suffix)
        path = [ifake, ireal]
        fid = evaluate_fid(path, gen_batch)
        return fid
    else:
        fid = []
        for i in range(5):
            ireal = os.path.join('../../Fidrealimg/'+opts.comb+'_'+suffix,'Label'+str(i))
            ifake = os.path.join(result_dir, 'Label' + str(i)+'_'+opts.suffix)
            path = [ifake, ireal]
            print(evaluate_fid(path,gen_batch))
            fid.append(evaluate_fid(path,gen_batch))
            # print(np.mean(fid))
        return fid,np.mean(fid)


def png_to_gif(png_path,gif_name):
    """png合成gif图像"""
    frames = []
    png_files = os.listdir(png_path)
    print(png_files)
    for frame_id in range(1,len(png_files)+1):
        frame = Image.open(os.path.join(png_path,'image%d.png'%frame_id))
        frames.append(frame)
    frames[0].save(gif_name,save_all=True,append_images=frames[1:],transparency=0,duration=2000,loop=0,disposal=2)

def store_display(opts,ep,real,fake,ssim=0,psnr=0,lpips=0,enDiv=False):
    fig = plt.figure(figsize=(15,10))
    if enDiv:
        string = 'Number = '+str(ep)+'       '+'LPIPS(mean) = %.4f'%lpips
        plt1Tiltle = 'Generated Image 1 '
        plt2Tiltle = 'Generated Image 2'
        saveName = 'num%d_Div_Img'%ep
        savedir = opts.gifdir + '/Divgif'
        if not os.path.exists(savedir):
            os.makedirs(savedir)
    else:
        string = 'Number = '+str(ep)+'       '+'SSIM(mean) = %.4f'%ssim+'       '+'PSNR(mean) = %.4f'%psnr
        plt1Tiltle = 'Ground Truth'
        plt2Tiltle = 'Generated Image'
        saveName = 'num%d_Fid_Img' % ep
        savedir = opts.gifdir + '/Fidgif'
        if not os.path.exists(savedir):
            os.makedirs(savedir)
    plt.text(0.5,
             0.9,
             string,
             c='black',
             weight='semibold',
             fontsize=15,
             verticalalignment="center",
             horizontalalignment="center",
             # bbox=dict(boxstyle='square',fc='white')
             )

    plt.axis('off')
    fig.add_subplot(121)
    plt.imshow(real)
    plt.title(plt1Tiltle,fontsize=15)
    fig.add_subplot(122)
    plt.imshow(fake)
    plt.title(plt2Tiltle,fontsize=15)
    plt.savefig(os.path.join(savedir,saveName))####
    plt.close()
