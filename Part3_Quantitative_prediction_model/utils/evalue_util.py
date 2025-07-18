
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from config.config_ import *
import math
import torch
import torch.nn.functional as F
from scipy.io import savemat, loadmat

# def multi_class_acc(preds, targs, th=1e-3):
#     pred = preds.cpu().detach().numpy().T
#     targ = targs.cpu().detach().numpy().T
#     idx_false = targ <= 0
#     idx_true = targ > 0
#     diff = (pred-targ)**2
#     diff_false = diff*idx_false*0.1
#     diff_true = diff*idx_true*0.9
#     diff_ = np.mean(diff_false+diff_true,0)
#     return (diff_ <= th).mean()


# def multi_class_acc(preds, targs, th=10e-2):
#     pred = preds.cpu().detach().numpy().T
#     targ = targs.cpu().detach().numpy().T
#     diff = np.abs(pred-targ)
#     diff_ = np.all(diff <= th,axis=0)
#     return diff_.mean(),diff_

# def multi_class_acc(preds, targs, th=10e-2):
#     pred = preds.cpu().detach().numpy()
#     targ = targs.cpu().detach().numpy()
#     diff_sum = []
#     for i in range(len(pred)):
#         tmp = targ[i] > 0
#         sum_idx = np.sum(tmp)
#         diff = np.abs(pred[i]-targ[i])
#         if sum_idx == 2:
#             if np.all(diff[np.where(tmp==1)[0]] <= 0.15,axis=0):
#                 diff_ = np.all(diff[np.where(tmp==0)[0]] <= 0.05,axis=0)
#             else:
#                 diff_ = np.all(diff[np.where(tmp==1)[0]] <= 0.15,axis=0)
#         elif sum_idx == 1 :
#             diff_ = np.all(diff <= 0.05,axis=0)
#         diff_sum.append(diff_)
#     return np.array(diff_sum).mean(),np.array(diff_sum)

def multi_class_acc(preds, targs, th=10e-2):
    pred = preds.cpu().detach().numpy()
    sum_ = np.sum(pred, axis=1).reshape([-1, 1])
    pred = np.divide(pred, np.concatenate([sum_, sum_, sum_, sum_, sum_], axis=1))
    targ = targs.cpu().detach().numpy()
    diff_sum = []
    for i in range(len(pred)):
        flag = 0
        for n in range(NUM_CLASSES):
            _pred = pred[i,n]
            _targ = targ[i,n]
            # if _targ == 0 :
            #     if _pred <= 0.05:
            #         flag +=1
            # if _targ == 0.25:
            #     if _pred > 0.05 and _pred <= 0.375:
            #         flag += 1
            # if _targ == 0.5:
            #     if _pred > 0.375 and _pred <= 0.625:
            #         flag += 1
            # if _targ == 0.75:
            #     if _pred > 0.625 and _pred <= 0.95:
            #         flag += 1
            # if _targ == 1:
            #     if _pred > 0.95 and _pred <= 1:
            #         flag += 1
            if _targ == 0:
                if _pred < 0.1:
                    flag += 1
            if _targ == 0.25:
                if _pred >= 0.1 and _pred < 0.35:
                    flag += 1
            if _targ == 0.5:
                if _pred >= 0.35 and _pred < 0.65:
                    flag += 1
            if _targ == 0.75:
                if _pred >= 0.65 and _pred < 0.9:
                    flag += 1
            if _targ == 1:
                if _pred >= 0.9 and _pred <= 1.0:
                    flag += 1
        if flag == 5:
            diff_sum.append(1)
        else:
            diff_sum.append(0)
    return np.array(diff_sum).mean(),np.array(diff_sum)

# def multi_class_acc(preds, targs, th=10e-2):
#     pred = preds.cpu().detach().numpy()
#     sum_ = np.sum(pred, axis=1).reshape([-1, 1])
#     pred = np.divide(pred, np.concatenate([sum_, sum_, sum_, sum_, sum_], axis=1))
#     targ = targs.cpu().detach().numpy()
#     diff_sum = []
#     for i in range(len(pred)):
#         flag = 0
#         for n in range(NUM_CLASSES):
#             _pred = pred[i,n]
#             _targ = targ[i,n]
#             if _targ == 0 :
#                 if _pred < 0.1:
#                     flag +=1
#             if _targ == 0.25:
#                 if _pred >= 0.1 and _pred < 0.375:
#                     flag += 1
#             if _targ == 0.5:
#                 if _pred >= 0.375 and _pred < 0.625:
#                     flag += 1
#             if _targ == 0.75:
#                 if _pred >= 0.625 and _pred < 0.9:
#                     flag += 1
#             if _targ == 1:
#                 if _pred >= 0.9 and _pred <= 1:
#                     flag += 1
#         if flag == 5:
#             diff_sum.append(1)
#         else:
#             diff_sum.append(0)
#     return np.array(diff_sum).mean(),np.array(diff_sum)

# def pos_consistent(pred,targs):
#     _, idcs = torch.topk(pred, 2, dim=1, largest=True)
#     idcs = torch.sort(idcs, dim=1)[0]
#     for i


# def person_corr(preds, targs):
#     pred = preds.cpu().detach().numpy()
#     targ = targs.cpu().detach().numpy()
#     personCorr = []
#     for i in range(len(pred)):
#         if COMB == 3:
#             personCorr_ = np.corrcoef(pred[i,(1,3)], targ[i,(1,3)])[0,1]
#         elif COMB == 2:
#             personCorr_ = np.corrcoef(pred[i, (1, 2)], targ[i, (1, 2)])[0, 1]
#         elif COMB == 1:
#             personCorr_ = np.corrcoef(pred[i, (0, 4)], targ[i, (0, 4)])[0, 1]
#         elif COMB == 0:
#             personCorr_ = np.corrcoef(pred[i, (0, 1)], targ[i, (0, 1)])[0, 1]
#         if math.isnan(personCorr_):
#             personCorr_ = 0
#         personCorr.append(personCorr_)
#     personCorr = np.array(personCorr)
#     return personCorr.mean(),personCorr

# def pos_consistent(pred,index):
#     pred = pred.cpu().detach().numpy()
#     targ = loadmat(mat_posCon)
#     main_idx = targ['mainLoc'][0][index]
#     addi_idx = targ['additionLoc'][0][index]
#     acc_n = []
#     for i in range(len(pred)):
#         if COMB == 0:
#             targ0 = pred[i,(2,3,4)]
#             targ1 = pred[i,(0,1)]
#         elif COMB == 1:
#             targ0 = pred[i, (1,2, 3)]
#             targ1 = pred[i, (0, 4)]
#         elif COMB == 2:
#             targ0 = pred[i, (0, 3, 4)]
#             targ1 = pred[i, (1, 2)]
#         elif COMB == 3:
#             targ0 = pred[i, (0, 2, 4)]
#             targ1 = pred[i, (1, 3)]
#         else:
#             raise ValueError
#         if (targ0 <= 0.01).all() and (targ1>=0.01).all():
#             main_ = pred[i,main_idx[i]]
#             addi_ = pred[i,addi_idx[i]]
#             if main_ > addi_:
#                 acc_n.append(1)
#             else:
#                 acc_n.append(0)
#         else:
#             acc_n.append(0)
#     return np.array(acc_n).mean(), np.array(acc_n)

def pos_consistent(pred,index):
    pred = pred.cpu().detach().numpy()
    sum_ = np.sum(pred, axis=1).reshape([-1, 1])
    pred = np.divide(pred, np.concatenate([sum_, sum_, sum_, sum_, sum_], axis=1))
    targ = loadmat(mat_posCon)
    main_idx = targ['mainLoc'][0][index]
    addi_idx = targ['additionLoc'][0][index]
    acc_n = []
    for i in range(len(pred)):
        if COMB == 0:
            targ0 = pred[i,(2,3,4)]
            targ1 = pred[i,(0,1)]
        elif COMB == 1:
            targ0 = pred[i, (1,2, 3)]
            targ1 = pred[i, (0, 4)]
        elif COMB == 2:
            targ0 = pred[i, (0, 3, 4)]
            targ1 = pred[i, (1, 2)]
        elif COMB == 3:
            targ0 = pred[i, (0, 2, 4)]
            targ1 = pred[i, (1, 3)]
        else:
            raise ValueError
        min_ = np.min(targ1)
        if ((targ0 - min_)<0.001).all():
            main_ = pred[i,main_idx[i]]
            addi_ = pred[i,addi_idx[i]]
            if main_ > addi_:
                acc_n.append(1)
            else:
                acc_n.append(0)
        else:
            acc_n.append(0)
    return np.array(acc_n).mean(), np.array(acc_n)

# def pos_consistent(pred,index):
#     pred = pred.cpu().detach().numpy()
#     targ = loadmat(mat_posCon)
#     main_idx = targ['mainLoc'][0][index]
#     addi_idx = targ['additionLoc'][0][index]
#     acc_n = []
#     for i in range(len(pred)):
#         if COMB == 0:
#             targ0 = pred[i,(2,3,4)]
#             targ1 = pred[i,(0,1)]
#         elif COMB == 1:
#             targ0 = pred[i, (1,2, 3)]
#             targ1 = pred[i, (0, 4)]
#         elif COMB == 2:
#             targ0 = pred[i, (0, 3, 4)]
#             targ1 = pred[i, (1, 2)]
#         elif COMB == 3:
#             targ0 = pred[i, (0, 2, 4)]
#             targ1 = pred[i, (1, 3)]
#         else:
#             raise ValueError
#         main_ = pred[i,main_idx[i]]
#         addi_ = pred[i,addi_idx[i]]
#         if main_ > addi_:
#             acc_n.append(1)
#         else:
#             acc_n.append(0)
#     return np.array(acc_n).mean(), np.array(acc_n)

# def pos_consistent(pred,index):
#     pred = pred.cpu().detach().numpy()
#     targ = loadmat(mat_posCon)
#     main_idx = targ['mainLoc'][0][index]
#     addi_idx = targ['additionLoc'][0][index]
#     acc_n = []
#     for i in range(len(pred)):
#         main_ = pred[i,main_idx[i]]
#         addi_ = pred[i,addi_idx[i]]
#         if main_ > addi_:
#             acc_n.append(1)
#         else:
#             acc_n.append(0)
#     return np.array(acc_n).mean(), np.array(acc_n)

def person_corr(preds, targs):
    pred = preds.cpu().detach().numpy()
    sum_ = np.sum(pred, axis=1).reshape([-1, 1])
    pred = np.divide(pred, np.concatenate([sum_, sum_, sum_, sum_, sum_], axis=1))
    targ = targs.cpu().detach().numpy()
    personCorr = []
    for i in range(len(pred)):
        personCorr_ = np.corrcoef(pred[i,:], targ[i,:])[0,1]
        if math.isnan(personCorr_):
            personCorr_ = 0
        personCorr.append(personCorr_)
    personCorr = np.array(personCorr)
    return personCorr.mean(),personCorr

def mse_(preds, targs):
    pred = preds.cpu().detach().numpy()
    sum_ = np.sum(pred, axis=1).reshape([-1, 1])
    pred = np.divide(pred, np.concatenate([sum_, sum_, sum_, sum_, sum_], axis=1))
    targ = targs.cpu().detach().numpy()
    diff_ = mse(pred.T,targ.T,multioutput='raw_values')
    return diff_.mean(),diff_


# def multi_class_acc(preds, targs, th=5e-2):
#     pred = preds.cpu().detach().numpy()
#     targ = targs.cpu().detach().numpy()
#     diff = np.abs(pred[:,CATE]-targ[:,CATE])
#     diff_ = diff <= th
#     return diff_.mean(),diff_
#
# def person_corr(preds, targs):
#     pred = preds.cpu().detach().numpy()
#     targ = targs.cpu().detach().numpy()
#     personCorr = []
#     for i in range(len(pred)):
#         personCorr_ = np.corrcoef(pred[i,CATE], targ[i,CATE])[0,1]
#         personCorr.append(personCorr_)
#     personCorr = np.array(personCorr)
#     return personCorr.mean(),personCorr
#
# def mse_(preds, targs):
#     pred = preds.cpu().detach().numpy()[:,CATE]
#     targ = targs.cpu().detach().numpy()[:,CATE]
#     diff_ = (pred-targ)**2
#     return diff_.mean(),diff_

# def sing_acc(preds, targs):
