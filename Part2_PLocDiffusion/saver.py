import os
import cv2
from imageio import imsave, imread
import numpy as np
import evaluate_model as evalmodel


def save_image(args, imgs, inputs, epoch,imgs_test_folder=None):
    height = args.height
    batch_size = len(imgs)
    if imgs_test_folder == None:
        imgs_test_folder = args.img_out_dir
    if not os.path.exists(imgs_test_folder):
        os.makedirs(imgs_test_folder)

    temp_test_dir = os.path.join(imgs_test_folder, 'epoch_%d_#img.png' % (epoch))

    res = np.zeros((height * batch_size + 2 * (batch_size - 1), height * 3 + 4, 3))

    for k in range(batch_size):
        res[height * k + 2 * k:height * (k + 1) + 2 * k, 0:height, 2] = (np.clip((inputs[k, :, :, 0] + 1) * 127.5, 0, 255)).astype(np.uint8)
        res[height * k + 2 * k:height * (k + 1) + 2 * k, 0:height, 0] = (np.clip((inputs[k, :, :, 2] + 1) * 127.5, 0, 255)).astype(np.uint8)
        res[height * k + 2 * k:height * (k + 1) + 2 * k,height + 2:height * 2 + 2, 1] = \
            (np.clip((inputs[k, :, :, 1] + 1) * 127.5, 0, 255)).astype(np.uint8)
        res[height * k + 2 * k:height * (k + 1) + 2 * k,height * 2 + 4:height * 3 + 4, 1] = \
            (np.clip((imgs[k, :, :, 0] + 1) * 127.5, 0, 255)).astype(np.uint8)
    imsave(temp_test_dir, res.astype(np.uint8))


def write_logs(writer,total_it,savelist):
    for i in savelist.keys():
        writer.add_scalar('/'+i, savelist[i], total_it)


def evaluate(inputs,pred, labels, imgs, thr=1e-3):
    imgs = imgs.permute(0, 2, 3, 1).cpu().detach().numpy()
    inputs = inputs.permute(0, 2, 3, 1).cpu().detach().numpy()
    _, ssim = evalmodel.cal_ssim(inputs, imgs, labels,singley=False)
    _, psnr = evalmodel.cal_psnr(inputs, imgs, labels,singley=False)
    _, labelAcc = evalmodel.labelAcc(pred, labels, thr, len(labels))
    return ssim, psnr,labelAcc








