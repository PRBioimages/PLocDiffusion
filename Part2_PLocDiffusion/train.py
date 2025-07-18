import  os
my_env = os.environ.copy()
#my_env["PATH"] = "/home/liyu/software/Anaconda3/envs/liyuDM/bin:" + my_env["PATH"]
os.environ.update(my_env)
import sys
sys.path.insert(0, '..')
import torch

from torch.autograd.variable import Variable
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.nn import DataParallel
import time
from tensorboardX import SummaryWriter

from options import TrainOptions
from model import PLocDiffusion
from data_reader import data_reader

from saver import evaluate, save_image, write_logs
from log_util import Logger
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
from pytorch_wavelets import DWTForward, DWTInverse
from diffusion import Posterior_Coefficients, Diffusion_Coefficients
osp = os.path
ope = os.path.exists
opj = os.path.join
def main():
    torch.autograd.set_detect_anomaly = True
    parser = TrainOptions()
    args = parser.parse()
    log_out_dir = opj('../logs', 'log' + args.out_dir)
    if not ope(log_out_dir):
        os.makedirs(log_out_dir)
    log = Logger()
    log.open(opj(log_out_dir, 'log.train.txt'), mode='a')

    model_out_dir = opj('../models', 'model' + args.out_dir)
    args.modeldir = model_out_dir
    log.write(">> Creating directory if it does not exist:\n>> '{}'\n".format(model_out_dir))
    if not ope(model_out_dir):
        os.makedirs(model_out_dir)

    imgs_out_dir = opj('../imgs', 'imgs' + args.out_dir)
    args.img_out_dir = imgs_out_dir
    if not ope(imgs_out_dir):
        os.makedirs(imgs_out_dir)

    logtf_train_out_dir = opj('../logs', 'logtf'+ args.out_dir, 'train')
    if not ope(logtf_train_out_dir):
        os.makedirs(logtf_train_out_dir)
    writer_train = SummaryWriter(log_dir=logtf_train_out_dir)

    logtf_val_out_dir = opj('../logs', 'logtf' + args.out_dir, 'val')
    if not ope(logtf_val_out_dir):
        os.makedirs(logtf_val_out_dir)
    writer_val = SummaryWriter(log_dir=logtf_val_out_dir)


    args.ori_image_size = args.image_size
    args.image_size = args.current_resolution
    start_epoch = 0
    global_step = 0
    # set random seeds
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    model = PLocDiffusion(args)
    model.setgpu()

    log.write("creating data loader...")
    # saver for display and output
    train_dataset = data_reader(is_Train=True)
    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    valid_dataset = data_reader(is_Train=False)
    valid_loader = DataLoader(
        valid_dataset,
        sampler=SequentialSampler(valid_dataset),
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True
    )

    if args.resume:
        args.resume = os.path.join(model_out_dir, args.resume)
        if os.path.isfile(args.resume):
            log.write(">> Loading checkpoint:\n>> '{}'\n".format(args.resume))
            start_epoch,global_step = model.resume(args.resume)
            optimizer_fpath = args.resume.replace('.pth', '_optim.pth')
            if ope(optimizer_fpath):
                log.write(">> Loading checkpoint:\n>> '{}'\n".format(optimizer_fpath))
                model.optim_resume(optimizer_fpath)
            log.write(">>>> loaded checkpoint:\n>>>> '{}' (epoch {})\n".format(args.resume, start_epoch))
        else:
            log.write(">> No checkpoint found at '{}'\n".format(args.resume))


    log.write("training generate model...")
    # train
    log.write(
        '\nepoch    iter      lr_gen     lr_dis     |  gen train     val   |  dis train     val    |  min \n')
    log.write(
        '--------------------------------------------------------------------------------------------------------------------------\n')
    start_epoch += 1
    if not args.use_pytorch_wavelet:
        dwt = DWT_2D("haar")
        iwt = IDWT_2D("haar")
    else:
        dwt = DWTForward(J=1, mode='zero', wave='haar').cuda()
        iwt = DWTInverse(mode='zero', wave='haar').cuda()
    device = torch.device('cuda:{}'.format('0'))
    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    for ep in range(start_epoch, args.max_epoch):
        np.random.seed(ep)
        torch.manual_seed(ep)
        torch.cuda.manual_seed_all(ep)
        end = time.time()

        iter,losses,global_step = train_(train_loader, model, ep,args,dwt,iwt,coeff,pos_coeff,global_step)
        # Save network weights
        model.save(model_out_dir, ep, global_step)

        lr_gen,lr_dis = model.gen_opt.param_groups[0]['lr']\
                       , model.dis_opt.param_groups[0]['lr']
        train_dict = losses
        with torch.no_grad():
            valid_dict = val_(valid_loader, model,dwt,iwt,coeff,pos_coeff,save_iter=100)

        print('\r', end='', flush=True)
        # save valid loss
        write_logs(writer_train,(ep + 1) * iter + 1,train_dict)
        write_logs(writer_val,(ep + 1) * iter + 1,valid_dict)
        print('\r', end='', flush=True)
        log.write(
            '%5.1f   %5d    %0.6f   %0.6f   |  %0.4f    %0.4f     |  %0.4f    %0.4f      | %3.1f min \n' % \
            (ep, iter + 1, lr_gen,lr_dis, train_dict['g_loss'], valid_dict['g_loss'], \
                  train_dict['d_loss'], valid_dict['d_loss'],(time.time() - end) / 60))


def train_(train_loader, model,epoch,args,dwt,iwt,coeff,pos_coeff,global_step):
    model.train()
    sum_iter = len(train_loader)
    print_freq = 1
    losses = {'g_loss_content': AverageMeter(),
              'g_loss_label': AverageMeter(),
              'g_loss_adv': AverageMeter(),
              'g_loss': AverageMeter(),
              'd_loss': AverageMeter(),
              'd_loss_adv': AverageMeter(),
              'd_loss_real_adv': AverageMeter(),
              'd_loss_fake_adv': AverageMeter(),
              'd_loss_real_label': AverageMeter(),
              'd_loss_fake_label': AverageMeter(),
              'd_loss_label': AverageMeter(),
              }

    for iter, iter_data in enumerate(train_loader, 0):
        inputs, labels = iter_data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        # update model
        loss,global_step = model.update(epoch, iter, sum_iter, inputs, labels,dwt,iwt,coeff,pos_coeff,global_step)
        for ikey in losses.keys():
            losses[ikey].update(loss[ikey], n=len(labels))
        if (iter + 1) % print_freq == 0 or iter == 0 or (iter + 1) == sum_iter:
            print('\r%5.1f   %5d    %0.6f    %0.6f|  %0.4f   %0.4f  | ... ' % \
                  (epoch - 1 + (iter + 1) / sum_iter, iter + 1, model.gen_opt.param_groups[0]['lr']\
                       , model.dis_opt.param_groups[0]['lr'], losses['g_loss'].avg, losses['d_loss'].avg),end='', flush=True)

    dict_ = {}
    for ikey in losses.keys():
        dict_.update({ikey: losses[ikey].avg})
    return iter,dict_,global_step


def val_(validsyn_loader, model,epoch,args,dwt,iwt,coeff,pos_coeff,save_iter):
    Acc = AverageMeter()
    psnr = AverageMeter()
    ssim = AverageMeter()
    losses = {'g_loss_content': AverageMeter(),
              'g_loss_label': AverageMeter(),
              'g_loss_adv': AverageMeter(),
              'g_loss': AverageMeter(),
              'd_loss': AverageMeter(),
              'd_loss_adv': AverageMeter(),
              'd_loss_real_adv': AverageMeter(),
              'd_loss_fake_adv': AverageMeter(),
              'd_loss_real_label': AverageMeter(),
              'd_loss_fake_label': AverageMeter(),
              'd_loss_label': AverageMeter(),
              }
    model.eval()
    end = time.time()
    for iter, iter_data in enumerate(validsyn_loader, 0):
        inputs, labels = iter_data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        loss,pred,generated = model.test_forward(inputs, labels,dwt,iwt,coeff,pos_coeff,save_imag = True)
        validssim, validpsnr, validlabelAcc = evaluate(inputs,pred, labels, generated)
        for ikey in losses.keys():
            losses[ikey].update(loss[ikey], n=len(labels))
        Acc.update(validlabelAcc, n=len(labels))
        psnr.update(validpsnr, n=len(labels))
        ssim.update(validssim, n=len(labels))
        if iter > save_iter:
            inputs = inputs.permute(0, 2, 3, 1).cpu().detach().numpy()
            generated = generated.permute(0, 2, 3, 1).cpu().detach().numpy()
            save_image(args, generated, inputs, epoch)
            break
    print('\nSample time for one image = %.2fs'%((time.time()-end)/iter))
    dict_ = {'psnr':psnr.avg,'ssim':ssim.avg,'acc':Acc.avg}
    for ikey in losses.keys():
        dict_.update({ikey:losses[ikey].avg})
    return dict_

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
