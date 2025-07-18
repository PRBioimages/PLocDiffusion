import  os
my_env = os.environ.copy()
#my_env["PATH"] = "/home/liyu/software/Anaconda3/envs/liyuDM/bin:" + my_env["PATH"]
os.environ.update(my_env)
import sys
sys.path.insert(0, '..')
import torch
import cv2
import evaluate_model
from PIL import Image
from torch.autograd.variable import Variable
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.nn import DataParallel
import time
from tensorboardX import SummaryWriter

from options import TestOptions
from model import PLocDiffusion
from data_reader_test import data_reader

from saver import evaluate,write_logs
from log_util import Logger
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
from pytorch_wavelets import DWTForward, DWTInverse
from diffusion import Posterior_Coefficients, Diffusion_Coefficients
import time
osp = os.path
ope = os.path.exists
opj = os.path.join


def evaluate_ssim(args,data,model,dwt,iwt,coeff,pos_coeff):
    ssim = []
    iter = 0
    for iter, iter_data in enumerate(data, 0):
        torch.cuda.empty_cache()
        print('epoch:%d' % iter)
        if iter > 199:
            break
        inputs, labels = iter_data
        generated = np.zeros((len(inputs), 256, 256, 1), dtype='float32')
        for i in range(1):
            # generated = np.zeros((len(inputs), 256, 256, 1), dtype='float32')
            for n in range(len(inputs)):
                generated[n, :, :, 0] = cv2.imread(('../GenDM/Sum_U2La_/' + 'epoch_%d_batch_%d_#img_%d.png') % (iter, n, i))[:, :, 1]
        inputs = inputs.permute(0, 2, 3, 1)
        inputs = inputs.cpu().detach().numpy()
        inputs = (np.clip((inputs + 1) * 127.5, 0, 255)).astype(np.uint8)
        tmpssim, _ = evaluate_model.cal_ssim(inputs, generated, labels.cpu().detach().numpy(), singley=False)
        ssim += tmpssim
        iter = iter + 1
        print(np.mean(np.array(ssim)))

    return np.mean(np.array(ssim))

def evaluate_psnr(args,data,model,dwt,iwt,coeff,pos_coeff):
    psnr = []
    iter = 0
    for iter, iter_data in enumerate(data, 0):
        torch.cuda.empty_cache()
        print('epoch:%d' % iter)
        if iter > 199:
            break
        inputs, labels = iter_data
        for i in range(1):
            generated = np.zeros((len(inputs), 256, 256, 1), dtype='float32')
            for n in range(len(inputs)):
                generated[n, :, :, 0] = cv2.imread(('../GenDM/Sum_U2La_/' + 'epoch_%d_batch_%d_#img_%d.png') % (iter, n, i))[:, :, 1]
        inputs = inputs.permute(0, 2, 3, 1)
        inputs = inputs.cpu().detach().numpy()
        inputs = (np.clip((inputs + 1) * 127.5, 0, 255)).astype(np.uint8)
        tmppsnr, _ = evaluate_model.cal_psnr(inputs, generated, labels.cpu().detach().numpy(), singley=False)
        psnr += tmppsnr
        iter = iter + 1
        print(np.mean(np.array(psnr)))

    return np.mean(np.array(psnr))


def find_epoch_ssim(i,args,model,dwt,iwt,coeff,pos_coeff,dataloader):
    # model
    print('\n--- load model ---', i)
    model.eval()
    resume = os.path.join(args.model_out_dir, i)
    _, _ = model.resume(resume, train=False)

    # directory
    result_dir = args.img_dir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # test
    print('\n--- testing ---')

    ssim = evaluate_ssim(args,dataloader, model,dwt,iwt,coeff,pos_coeff)


    return ssim


def find_epoch_psnr(i,args,model,dwt,iwt,coeff,pos_coeff,dataloader):
    # model
    print('\n--- load model ---', i)
    model.eval()
    resume = os.path.join(args.model_out_dir, i)
    _, _ = model.resume(resume, train=False)

    # directory
    result_dir = args.img_dir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    print('\n--- testing ---')

    psnr = evaluate_psnr(args,dataloader, model,dwt,iwt,coeff,pos_coeff)

    return psnr


def evaluate_diversity(opts,data,model,dwt,iwt,coeff,pos_coeff,result_dir,mode='train'):
    fid = 0
    generate_img_z(opts, data, model,dwt,iwt,coeff,pos_coeff, result_dir,mode)
    fid = evaluate_model.calc_fid(opts, result_dir, suffix=opts.suffix, ensum=True)

    return fid


def evaluate_lpips(opts,data,model):
    # # calculate LPIPS
    # # Ref: https://github.com/richzhang/PerceptualSimilarity
    genImg_lpips = evaluate_model.calc_lpips(opts,data,model)
    print('LPIPS:%.4f' % genImg_lpips)

def generate_img_z(args,data,model,dwt,iwt,coeff,pos_coeff,path,mode='train'):
    sum_store_path = os.path.join(path, 'Sum_'+args.suffix)
    if not os.path.exists(sum_store_path):
        os.makedirs(sum_store_path)
    i = 0
    end = time.time()
    for iter, iter_data in enumerate(data, 0):
        if iter > 999:
            break
        inputs, labels = iter_data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        for k in range(10):
            generated = model.test_(inputs, labels,dwt,iwt,pos_coeff)
            generated = generated.permute(0, 2, 3, 1).cpu().detach().numpy()
            for n in range(len(labels)):# 意思是 for n in range（5）：
                res = np.zeros((256, 256, 3))
                res[:, :, 1] = (np.clip((generated[n, :, :, 0] + 1) * 127.5, 0, 255))
                cv2.imwrite(os.path.join(sum_store_path, 'epoch_%d_batch_%d_#img_%d.png') % (iter, n, k),
                            (res).astype(np.uint8))
            i = i+1
    print('\nSample time for one image = %.2fs' % ((time.time() - end) / i))


def find_epoch(i,args,model,dwt,iwt,coeff,pos_coeff,dataloader):
    # model
    print('\n--- load model ---', i)
    model.eval()
    resume = os.path.join(args.model_out_dir, i)
    _,_ = model.resume(resume, train=False)

    # directory
    result_dir = args.img_dir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # test
    print('\n--- testing ---')
    fid = evaluate_diversity(args,dataloader,model,dwt,iwt,coeff,pos_coeff,result_dir,mode='test')
    return fid

def main():
    parser = TestOptions()
    args = parser.parse()

    log_out_dir = opj('../logs', 'log' + args.out_dir)
    if not ope(log_out_dir):
        os.makedirs(log_out_dir)
    log = Logger()
    log.open(opj(log_out_dir, 'log.test.txt'), mode='a')
    model_out_dir = opj('../models', 'model' + args.out_dir)
    args.ori_image_size = args.image_size
    args.image_size = args.current_resolution
    model = PLocDiffusion(args)
    model.setgpu()
    model.eval()
    # Wavelet Pooling

    if not args.use_pytorch_wavelet:
        dwt = DWT_2D("haar")
        iwt = IDWT_2D("haar")
    else:
        dwt = DWTForward(J=1, mode='zero', wave='haar').cuda()
        iwt = DWTInverse(mode='zero', wave='haar').cuda()
    device = torch.device('cuda:{}'.format('0'))
    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)

    fids = []
    ep = []
    n = 0
    for i in os.listdir(model_out_dir):
        if 'optim' in i:
            continue
        iep = int(i.split('.pth')[0])
        if 17 < iep:
            continue
        args.model_out_dir = model_out_dir
        args.img_dir = opj('../GenDM', 'Gen' + args.out_dir)
        test_dataset = data_reader()
        test_loader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=args.workers,
            pin_memory=True
        )
        if n == 0:
            evaluate_model.diversity_genimg(args, test_loader, args.suffix)
            n = n+1
        fid = find_epoch(i,args,model,dwt,iwt,coeff,pos_coeff,test_loader)
        print('ep-----------%s' % i)
        fids.append(fid)
        ep.append(i)
        print(fids)
        print(ep)
        print(np.argmin(np.array(fids)))
        a = np.argmin(np.array(fids))
        print(ep[a])



if __name__ == "__main__":
    main()
