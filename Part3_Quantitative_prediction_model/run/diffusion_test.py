import argparse
import os
from Part2_PLocDiffusion.model import PLocDiffusion
from utils.common_util import *
from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
from pytorch_wavelets import DWTForward, DWTInverse
from Part2_PLocDiffusion.diffusion import Posterior_Coefficients, Diffusion_Coefficients
import torch
from options import TestOptions
import torch.nn as nn
from utils.tool import *


gan_modelnamelist = {'CyNp': 'CyNp.pth',
                     'CyPM': 'CyPM.pth',
                     'NuMi': 'NuMi.pth',
                     'NuNu': 'NuNu.pth'}


class ganmodel_(nn.Module): #这个是四种细胞图片组合
    def __init__(self):
        super(ganmodel_, self).__init__()
        parser = TestOptions()
        self.opts = parser.parse()
        self.opts.ori_image_size = self.opts.image_size
        self.opts.image_size = self.opts.current_resolution  # 这两行真的很重要
        if not self.opts.use_pytorch_wavelet:
            self.dwt = DWT_2D("haar")
            self.iwt = IDWT_2D("haar")
        else:
            self.dwt = DWTForward(J=1, mode='zero', wave='haar').cuda()
            self.iwt = DWTInverse(mode='zero', wave='haar').cuda()
        device = torch.device('cuda:{}'.format('0'))
        self.coeff = Diffusion_Coefficients(self.opts, device)
        self.pos_coeff = Posterior_Coefficients(self.opts, device)
        print('\n--- load gan model ---')
        self.model_cynp = PLocDiffusion(self.opts)
        self.model_cynp.eval()
        self.model_cynp.setgpu()
        model_dir_cynp = opj(self.opts.working_directory, gan_modelnamelist['CyNp'])
        print('Load Trained %s Model-%03d.pth' %('CyNp',checkpoint['CyNp']))
        self.model_cynp.resume(model_dir_cynp, train=False)

        self.model_cypm = PLocDiffusion(self.opts)
        self.model_cypm.eval()
        self.model_cypm.setgpu()
        model_dir_cypm = opj(self.opts.working_directory,  gan_modelnamelist['CyPM'])
        print('Load Trained %s Model-%03d.pth' %('CyPM',checkpoint['CyPM']))
        self.model_cypm.resume(model_dir_cypm, train=False)

        self.model_numi = PLocDiffusion(self.opts)
        self.model_numi.eval()
        self.model_numi.setgpu()
        model_dir_numi = opj(self.opts.working_directory, gan_modelnamelist['NuMi'])
        print('Load Trained %s Model-%03d.pth' %('NuMi',checkpoint['NuMi']))
        self.model_numi.resume(model_dir_numi, train=False)

        self.model_nunu = PLocDiffusion(self.opts)
        self.model_nunu.eval()
        self.model_nunu.setgpu()
        model_dir_nunu = opj(self.opts.working_directory,  gan_modelnamelist['NuNu'])
        print('Load Trained %s Model-%03d.pth' %('NuNu',checkpoint['NuNu']))
        self.model_nunu.resume(model_dir_nunu, train=False)


    def gan_test_(self,image,label_2,comb):

        if comb == 'NuMi'or comb == 'Mi':
            self.model = self.model_numi
        elif comb == 'NuNu'or comb == 'Nu':
            self.model = self.model_nunu
        elif comb == 'CyNp' or comb == 'Np' or comb == 'Cy':
            self.model = self.model_cynp
        elif comb == 'CyPM' or comb == 'PM':
            self.model = self.model_cypm
        elif comb == 'None':
            raise ValueError('comb can not be None!')
        image = image_to_tensor(image)
        label_2 = label_to_tensor(label_2)
        output = self.model.test_generate(image, label_2, self.dwt, self.iwt, self.pos_coeff)
        output = output.squeeze(0)
        return output



