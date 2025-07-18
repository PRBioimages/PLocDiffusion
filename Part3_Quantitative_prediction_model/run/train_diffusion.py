# coding: utf-8
import sys
sys.path.insert(0, '..')
import argparse
import time
import shutil
import warnings



import torch
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.nn import DataParallel
from torch.backends import cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from scipy.io import loadmat

from utils.common_util import *
from networks.imageclsnet import init_network
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from datasets.datareader_train_ import data_reader
# from classification.augment_util_gan import train_gan_augment
from layers.loss import *
from layers.scheduler import *
from utils.log_util import Logger
from utils.evalue_util import *


loss_names = ['FocalSymmetricLovaszHardLogLoss']

parser = argparse.ArgumentParser(description='PyTorch Protein Classification')
parser.add_argument('--out_dir', type=str,default='densenet_diffusion_Bat32_',
                    help='destination where trained network should be saved')

parser.add_argument('--gpu_id', default='0', type=str, help='gpu id used for training (default: 0)')
parser.add_argument('--num_classes', default=5, type=int, help='number of classes (default: 28)')
parser.add_argument('--in_channels', default=3, type=int, help='in channels (default: 4)')

parser.add_argument('--arch', default='class_densenet121_dropout', type=str,
                    help='model architecture (default: class_densenet121_dropout)')
parser.add_argument('--loss', default='FocalSymmetricLovaszHardLogLoss', choices=loss_names, type=str,
                    help='loss function: ' + ' | '.join(loss_names) + ' (deafault: FocalSymmetricLovaszHardLogLoss)')
parser.add_argument('--scheduler', default='Adam45', type=str, help='scheduler name')

parser.add_argument('--epochs', default=55, type=int, help='number of total epochs to run (default: 55)')
parser.add_argument('--batch_size', default=32, type=int, help='train mini-batch size (default: 32)')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 3)')

parser.add_argument('--clipnorm', default=1, type=int, help='clip grad norm')
parser.add_argument('--resume', default=0, type=str, help='name of the latest checkpoint (default: None)')

def main(REN):
    warnings.filterwarnings('ignore')
    args = parser.parse_args()

    log_out_dir = opj('../logs', 'log_'+args.out_dir+str(REN))#1，2，3，4
    if not ope(log_out_dir):
        os.makedirs(log_out_dir)
    log = Logger()
    log.open(opj(log_out_dir, 'log.train.txt'), mode='a')

    model_out_dir = opj('../models', 'model_'+args.out_dir+str(REN))
    log.write(">> Creating directory if it does not exist:\n>> '{}'\n".format(model_out_dir))
    if not ope(model_out_dir):
        os.makedirs(model_out_dir)

    # Use tensorboard to watch the training process
    logtf_train_out_dir = opj('../logs', 'logtf_' + args.out_dir+str(REN), 'train')
    if not ope(logtf_train_out_dir):
        os.makedirs(logtf_train_out_dir)
    writer_train = SummaryWriter(log_dir=logtf_train_out_dir)

    logtf_val_out_dir = opj('../logs', 'logtf_' + args.out_dir+str(REN), 'val')
    if not ope(logtf_val_out_dir):
        os.makedirs(logtf_val_out_dir)
    writer_val = SummaryWriter(log_dir=logtf_val_out_dir)

    logtf_valsyn_out_dir = opj('../logs', 'logtf_' + args.out_dir+str(REN), 'valsyn')
    if not ope(logtf_valsyn_out_dir):
        os.makedirs(logtf_valsyn_out_dir)
    writer_valsyn = SummaryWriter(log_dir=logtf_valsyn_out_dir)


    # set cuda visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cudnn.benchmark = True

    # set random seeds
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    model_params = {}
    model_params['architecture'] = args.arch
    model_params['num_classes'] = args.num_classes
    model_params['in_channels'] = args.in_channels
    model = init_network(model_params)

    # move network to gpu
    model = DataParallel(model)
    model.cuda()

    # define loss function (criterion)
    try:
        criterion = eval(args.loss)().cuda()
    except:
        raise(RuntimeError("Loss {} not available!".format(args.loss)))

    start_epoch = 0
    best_loss = 1e5
    best_epoch = 0

    # define scheduler
    try:
        scheduler = eval(args.scheduler)()
    except:
        raise (RuntimeError("Scheduler {} not available!".format(args.scheduler)))
    optimizer = scheduler.schedule(model, start_epoch, args.epochs)[0]

    if args.resume:
        args.resume = os.path.join(model_out_dir, args.resume)
        if os.path.isfile(args.resume):
            # load checkpoint weights and update model and optimizer
            log.write(">> Loading checkpoint:\n>> '{}'\n".format(args.resume))

            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_epoch = checkpoint['best_epoch']
            best_loss = checkpoint['best_score']
            model.module.load_state_dict(checkpoint['state_dict'])

            optimizer_fpath = args.resume.replace('.pth', '_optim.pth')
            if ope(optimizer_fpath):
                log.write(">> Loading checkpoint:\n>> '{}'\n".format(optimizer_fpath))
                optimizer.load_state_dict(torch.load(optimizer_fpath)['optimizer'])
            log.write(">>>> loaded checkpoint:\n>>>> '{}' (epoch {})\n".format(args.resume, checkpoint['epoch']))
        else:
            log.write(">> No checkpoint found at '{}'\n".format(args.resume))

    # Data loading code
    Images_Norm_params = {}
    if ope(Imgs_mean_std):
        matdata = loadmat(Imgs_mean_std)
        mean = matdata['Mean']
        std = matdata['Std']
    else:
        raise (FileNotFoundError("The file {} is not found!".format(Imgs_mean_std)))
    # mean = [0.074598, 0.050630, 0.050891, 0.076287]#rgby
    # std =  [0.122813, 0.085745, 0.129882, 0.119411]
    Images_Norm_params['in_channels'] = args.in_channels
    Images_Norm_params['mean'] = mean
    Images_Norm_params['std'] = std

    print('Time to dataloader')
    # train_transform = train_gan_augment
    train_dataset = data_reader(net='diffusion',Images_Norm_params=Images_Norm_params,is_Train=True,is_Syn=False)
    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    valid_dataset = data_reader(net='diffusion',Images_Norm_params=Images_Norm_params,is_Train=False,is_Syn=False)
    valid_loader = DataLoader(
        valid_dataset,
        sampler=SequentialSampler(valid_dataset),
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=True
    )
    validsyn_dataset = data_reader(net='diffusion', Images_Norm_params=Images_Norm_params,is_Train=False, is_Syn=True)
    validsyn_loader = DataLoader(
        validsyn_dataset,
        sampler=SequentialSampler(validsyn_dataset),
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=True
    )

    focal_loss = FocalLoss().cuda()
    log.write('** start training here! **\n')
    log.write('\n')
    log.write('epoch    iter      rate     |  train_loss/acc/corr/mse  | valid_loss/acc/corr/mse |   validsyn_loss/acc/corr/mse   |  best_epoch/best_loss  |  min \n')
    log.write('---------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
    start_epoch += 1
    for epoch in range(start_epoch, args.epochs + 1):
        end = time.time()

        # set manual seeds per epoch
        np.random.seed(epoch)
        torch.manual_seed(epoch)
        torch.cuda.manual_seed_all(epoch)

        # adjust learning rate for each epoch
        lr_list = scheduler.step(model, epoch, args.epochs)
        lr = lr_list[0]
        # lr = optimizer.param_groups[0]['lr']
        # train for one epoch on train set
        iter, train_loss, train_acc,train_corr,train_mse\
            = train(train_loader, model, criterion, optimizer, epoch, clipnorm=args.clipnorm, lr=lr)

        with torch.no_grad():
            valid_loss, valid_acc,val_corr,val_mse\
                = validate(valid_loader, model, criterion, epoch,focal_loss)
            validsyn_loss, validsyn_acc, valsyn_corr, valsyn_mse = validate_syn(validsyn_loader, model, criterion,
                                                                                epoch, focal_loss)
        is_best = validsyn_loss < best_loss
        best_epoch = epoch if is_best else best_epoch
        best_loss = validsyn_loss if is_best else best_loss

        print('\r', end='', flush=True)
        log.write(
            '%5.1f   %5d    %0.6f   |  %0.4f  %0.4f  %0.4f  %0.4f  |  %0.4f  %6.4f  %0.4f  %0.4f  |    %0.4f  %6.4f %0.4f  %0.4f  |  %6.1f    %6.4f   | %3.1f min \n' % \
            (epoch, iter + 1, lr, train_loss, train_acc,train_corr,train_mse,valid_loss, valid_acc,\
             val_corr,val_mse,validsyn_loss, validsyn_acc, valsyn_corr, valsyn_mse,best_epoch, best_loss, (time.time() - end) / 60))
        write_logs(writer_train,(epoch + 1) * iter + 1,train_loss, train_acc)
        write_logs(writer_val, (epoch + 1) * iter + 1, valid_loss, valid_acc)
        write_logs(writer_valsyn, (epoch + 1) * iter + 1, validsyn_loss, validsyn_acc)
        save_model(model, is_best, model_out_dir, optimizer=optimizer, epoch=epoch, best_epoch=best_epoch, best_acc=best_loss

def train(train_loader, model, criterion, optimizer, epoch,clipnorm=1, lr=1e-5):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    personCorr = AverageMeter()
    MSE_ = AverageMeter()

    # switch to train mode
    model.train()

    num_its = len(train_loader)
    end = time.time()
    iter = 0
    print_freq = 1
    for iter, iter_data in enumerate(train_loader, 0):
        # measure data loading time
        data_time.update(time.time() - end)

        # zero out gradients so we can accumulate new ones over batches
        optimizer.zero_grad()

        images, labels = iter_data

        images = Variable(images.cuda(),requires_grad=True)
        labels = Variable(labels.cuda(),requires_grad=True)

        outputs = model(images)
        loss = criterion(outputs, labels, epoch=epoch)

        losses.update(loss.item())
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), clipnorm)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        logits = outputs
        probs = F.softmax(logits,dim=1)
        acc,tmp = multi_class_acc(probs, labels)
        corr,_ = person_corr(probs, labels)
        mse,_ = mse_(probs, labels)
        accuracy.update(acc.item(), n=len(tmp))
        personCorr.update(corr.item(), n=len(tmp))
        MSE_.update(mse.item(), n=len(tmp))

        if (iter + 1) % print_freq == 0 or iter == 0 or (iter + 1) == num_its:
            print('\r%5.1f   %5d    %0.6f   |  %0.4f  %0.4f  %0.4f  %0.4f  | ... ' % \
                  (epoch - 1 + (iter + 1) / num_its, iter + 1, lr, losses.avg, accuracy.avg,\
                   personCorr.avg,MSE_.avg), \
                  end='', flush=True)

    return iter, losses.avg, accuracy.avg,personCorr.avg,MSE_.avg

def validate(valid_loader, model, criterion, epoch, focal_loss, threshold=0.5):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    personCorr = AverageMeter()
    MSE_ = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for it, iter_data in enumerate(valid_loader, 0):
        # print(it)
        images, labels = iter_data

        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

        outputs = model(images)
        loss = criterion(outputs, labels, epoch=epoch)

        logits = outputs
        probs = F.softmax(logits,dim=1)

        acc,_ = multi_class_acc(probs, labels)
        corr,_ = person_corr(probs, labels)
        mse,_ = mse_(probs, labels)

        losses.update(loss.item())
        accuracy.update(acc.item(), n=len(labels))
        personCorr.update(corr.item(), n=len(labels))
        MSE_.update(mse.item(), n=len(labels))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg, accuracy.avg,personCorr.avg,MSE_.avg

def validate_syn(valid_loader, model, criterion, epoch, focal_loss, threshold=0.5):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    personCorr = AverageMeter()
    MSE_ = AverageMeter()
    model.eval()

    end = time.time()
    for it, iter_data in enumerate(valid_loader, 0):
        # print(it)
        images, labels = iter_data

        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

        outputs = model(images)
        loss = criterion(outputs, labels, epoch=epoch)

        logits = outputs
        probs = F.softmax(logits,dim=1)

        acc,_ = multi_class_acc(probs, labels)
        corr,_ = person_corr(probs, labels)
        mse,_ = mse_(probs, labels)

        losses.update(loss.item())
        accuracy.update(acc.item(), n=len(labels))
        personCorr.update(corr.item(), n=len(labels))
        MSE_.update(mse.item(), n=len(labels))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg, accuracy.avg, personCorr.avg, MSE_.avg


def save_model(model, is_best, model_out_dir, optimizer=None, epoch=None, best_epoch=None, best_acc=None):
    if type(model) == DataParallel:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    if epoch >= 0:
        model_fpath = opj(model_out_dir, '%03d.pth' % epoch)
        torch.save({
            'save_dir': model_out_dir,
            'state_dict': state_dict,
            'best_epoch': best_epoch,
            'epoch': epoch,
            'best_score': best_acc,
        }, model_fpath)

        optim_fpath = opj(model_out_dir, '%03d_optim.pth' % epoch)
        if optimizer is not None:
            torch.save({
                'optimizer': optimizer.state_dict(),
            }, optim_fpath)

        if is_best:
            best_model_fpath = opj(model_out_dir, 'final.pth')
            shutil.copyfile(model_fpath, best_model_fpath)
            if optimizer is not None:
                best_optim_fpath = opj(model_out_dir, 'final_optim.pth')
                shutil.copyfile(optim_fpath, best_optim_fpath)
    else:
        if is_best:
            best_model_fpath = opj(model_out_dir, 'final.pth')
            torch.save({
                'save_dir': model_out_dir,
                'state_dict': state_dict,
                'best_epoch': best_epoch,
                'epoch': epoch,
                'best_score': best_acc,
            }, best_model_fpath)
            if optimizer is not None:
                best_optim_fpath = opj(model_out_dir, 'final_optim.pth')
                torch.save({
                    'optimizer': optimizer.state_dict(),
                }, best_optim_fpath)





def write_logs(writer,total_it,loss,acc): #(###)
    writer.add_scalar('/loss', loss, total_it)
    writer.add_scalar('/acc', acc, total_it)



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

class SumMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0

    def update(self,val):
        self.sum += val


if __name__ == '__main__':
    print('%s: calling main function ... \n' % os.path.basename(__file__))
    for i in range(1,6):
        main(i)
    print('\nsuccess!')

