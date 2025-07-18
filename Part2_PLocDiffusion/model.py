import os
import network as networks
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion import sample_from_model, sample_posterior,q_sample_pairs,\
    get_time_schedule
import torch.distributed as dist
import matplotlib.pylab as plt
import time

class PLocDiffusion(nn.Module):
    def __init__(self, opts):
        super(PLocDiffusion, self).__init__()
        # parameters
        lr = 0.0001
        self.nz = opts.hiddenz_size
        self.opt = opts
        self.class_num = opts.num_class
        self.Gen = networks.WaveletNCSNpp(opts)
        self.Dis = networks.Discriminator(nc=2 * self.opt.num_channels, ngf=self.opt.ngf,
                           t_emb_dim=self.opt.t_emb_dim,
                           act=nn.LeakyReLU(0.2), num_layers=self.opt.num_disc_layers)
        # self.scaler = GradScaler()
        self.gen_opt = torch.optim.Adam(self.Gen.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0001)
        self.dis_opt = torch.optim.Adam(self.Dis.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0001)
        self.gen_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.gen_opt,
                                                                        T_max=self.opt.lr_decay_iter, eta_min=self.opt.learning_rate)
        self.dis_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.dis_opt, \
                                                                        T_max=self.opt.d_lr_decay_iter, eta_min=self.opt.d_learning_rate)

        self.BCE_loss = torch.nn.BCELoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        self.device = torch.device('cuda:{}'.format('0'))

    def setgpu(self):
        self.Gen = torch.nn.DataParallel(self.Gen).cuda()
        self.Dis = torch.nn.DataParallel(self.Dis).cuda()



    def set_requires_grad(self, net, requires_grad=False):
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


    def grad_penalty_call(self, args, D_real, x_t):
        grad_real = torch.autograd.grad(
            outputs=D_real.sum(), inputs=x_t, create_graph=True
        )[0]
        grad_penalty = (
                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
        ).mean()

        grad_penalty = args.r1_gamma / 2 * grad_penalty
        grad_penalty.backward()


    def iwt_D(self,iwt,fake_sample):
        fake_sample *= 2
        if not self.opt.use_pytorch_wavelet:
            fake_sample = iwt(
                fake_sample[:, :1], fake_sample[:, 1:2], fake_sample[:, 2:3], fake_sample[:, 3:4])
        else:
            fake_sample = iwt((fake_sample[:, :1], [torch.stack(
                (fake_sample[:, 1:2], fake_sample[:, 2:3], fake_sample[:, 3:4]), dim=2)]))
        return fake_sample


    def forward(self, real_image, ref_image, coeff, pos_coeff, label):
        t = torch.randint(0, self.opt.num_timesteps,
                          (real_image.size(0),)).cuda()
        x_t, x_tp1 = q_sample_pairs(coeff, real_image, t)
        x_t.requires_grad = True
        # train with fake
        latent_z = torch.randn(real_image.size(0), self.opt.nz, device=self.device)
        x_0_predict = self.Gen(x_tp1.detach(), ref_image, t, latent_z, label)
        x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
        D_real, real_label = self.Dis(x_t, t, x_tp1.detach())
        pred_fake, fake_label = self.Dis(x_pos_sample, t, x_tp1.detach())
        return D_real, real_label, pred_fake, fake_label, x_0_predict

    def update_D(self, pred_score, pred_label, label=None, mode=True):

        if mode:
            target_label = label
            loss_D_GAN = F.softplus(-pred_score).mean()
        else:
            target_label = torch.zeros_like(pred_label)
            target_label = target_label.cuda()
            loss_D_GAN = F.softplus(pred_score).mean()
        loss_D_label = self.compute_mse_loss(pred_label, target_label)

        return loss_D_GAN, loss_D_label


    def update_G(self,pred_fake,x_0_predict,real_image,fake_label,label):
        loss_G_GAN = F.softplus(-pred_fake.detach()).mean()
        loss_content = F.l1_loss(x_0_predict, real_image)
        loss_G_label = self.compute_mse_loss(fake_label.detach(), label)
        return loss_G_GAN,loss_content,loss_G_label


    def compute_mse_loss(self,fake,real):
        return torch.nn.functional.mse_loss(input=fake,target=real)

    def update(self, ep, it, sum_iter, image, label,dwt,iwt,coeff,pos_coeff,global_step):
        end1 = time.time()
        num_levels = int(np.log2(self.opt.ori_image_size // self.opt.current_resolution))
        iter = ep * sum_iter + it + 1
        # update discriminator
        self.set_requires_grad(self.Dis, True)
        self.dis_opt.zero_grad()
        realimage = image[:, 1, :, :].view([-1,1,256,256])
        refimage = image[:, (0,2), :, :]
        if not self.opt.use_pytorch_wavelet:
            for i in range(num_levels):
                xll, xlh, xhl, xhh = dwt(realimage)
                xll_ref, xlh_ref, xhl_ref, xhh_ref = dwt(refimage)
        else:
            xll, xh = dwt(realimage)
            xll_ref, xh_ref = dwt(refimage)
            xlh, xhl, xhh = torch.unbind(xh[0], dim=2)
            xlh_ref, xhl_ref, xhh_ref = torch.unbind(xh_ref[0], dim=2)

        real_data = torch.cat([xll, xlh, xhl, xhh], dim=1)
        ref_data = torch.cat([xll_ref, xlh_ref, xhl_ref, xhh_ref], dim=1)

        real_data = real_data / 2.0
        ref_data = ref_data/2.0

        assert -1 <= real_data.min() < 0
        assert 0 < real_data.max() <= 1

        assert -1 <= ref_data.min() < 0
        assert 0 < ref_data.max() <= 1

        real_image = real_data
        ref_image = ref_data


        D_real, real_label,pred_fake, fake_label,x_0_predict = self.forward(real_image,ref_image,coeff,pos_coeff,label)
        loss_D_real_GAN, loss_D_real_label = self.update_D(D_real, real_label,label)
        loss_D_fake_GAN,loss_D_fake_label= self.update_D(pred_fake, fake_label,mode=False)
        loss_D_GAN = loss_D_real_GAN+loss_D_fake_GAN
        loss_D_label = loss_D_real_label+loss_D_fake_label
        loss_D = loss_D_GAN+loss_D_label

        loss_D.backward(retain_graph=True)
        self.dis_opt.step()
        if iter <= self.opt.d_lr_decay_iter:
            self.dis_scheduler.step()

        ## Generator
        self.set_requires_grad(self.Dis, False)
        # update generator
        self.gen_opt.zero_grad()
        loss_G_GAN,loss_content,loss_G_label = self.update_G(pred_fake,x_0_predict,real_image,fake_label,label)
        loss_G = loss_G_GAN * self.opt.gamma_genL1 + loss_G_label * self.opt.gamma_genLabel + loss_content * self.opt.gamma_genMSE
        loss_G.backward()
        self.gen_opt.step()
        if iter <= self.opt.lr_decay_iter:
            self.gen_scheduler.step()
        global_step +=1
        losses = {'g_loss_content':loss_content.detach(),
                  'g_loss_label':loss_G_label.detach(),
                  'g_loss_adv':loss_G_GAN.detach(),
                  'g_loss':loss_G.detach(),
                  'd_loss':loss_D.detach(),
                  'd_loss_adv':loss_D_GAN.detach(),
                  'd_loss_real_adv':loss_D_real_GAN.detach(),
                  'd_loss_fake_adv':loss_D_fake_GAN.detach(),
                  'd_loss_real_label':loss_D_real_label.detach(),
                  'd_loss_fake_label':loss_D_fake_label.detach(),
                  'd_loss_label':loss_D_label.detach(),
                  }
        return losses,global_step

    def resume(self, model_dir, train=True):
        checkpoint = torch.load(model_dir)
        # weight
        if train:
            self.Dis.load_state_dict({k: v for k, v in checkpoint['dis'].items()})
        self.Gen.load_state_dict({k: v for k, v in checkpoint['gen'].items()})
        return checkpoint['epoch'],checkpoint['global_step']

    def optim_resume(self, model_dir, train=True):
        checkpoint = torch.load(model_dir)
        # optimizer
        if train:
            self.dis_opt.load_state_dict(checkpoint['dis_opt'])
            for state in self.dis_opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            self.gen_opt.load_state_dict(checkpoint['gen_opt'])
            for state in self.gen_opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            self.dis_scheduler.load_state_dict(checkpoint['dis_sch'])
            self.gen_scheduler.load_state_dict(checkpoint['gen_sch'])

    def save(self, model_out_dir, ep, global_step):
        model_fpath = os.path.join(model_out_dir, '%03d.pth' % ep)
        torch.save({
            'save_dir': model_out_dir,
            'dis': self.Dis.state_dict(),
            'gen': self.Gen.state_dict(),
            'epoch': ep,
            'global_step':global_step,
        }, model_fpath)

        optim_state = {'dis_opt': self.dis_opt.state_dict(),
                'gen_opt': self.gen_opt.state_dict(),
                'dis_sch': self.dis_scheduler.state_dict(),
                'gen_sch': self.gen_scheduler.state_dict()}

        optim_fpath = os.path.join(model_out_dir, '%03d_optim.pth' % ep)
        torch.save(optim_state, optim_fpath)
        return

    def test_forward(self, image, label,dwt,iwt,coeff,pos_coeff,save_imag = False):
        T = get_time_schedule(self.opt, self.device )

        num_levels = int(np.log2(self.opt.ori_image_size // self.opt.current_resolution))
        realimage = image[:, 1, :, :].view([-1, 1, 256, 256])
        refimage = image[:, (0, 2), :, :]
        if not self.opt.use_pytorch_wavelet:
            for i in range(num_levels):
                xll, xlh, xhl, xhh = dwt(realimage)
                xll_ref, xlh_ref, xhl_ref, xhh_ref = dwt(refimage)
        else:
            xll, xh = dwt(realimage)
            xll_ref, xh_ref = dwt(refimage)
            xlh, xhl, xhh = torch.unbind(xh[0], dim=2)
            xlh_ref, xhl_ref, xhh_ref = torch.unbind(xh_ref[0], dim=2)

        real_data = torch.cat([xll, xlh, xhl, xhh], dim=1)
        ref_data = torch.cat([xll_ref, xlh_ref, xhl_ref, xhh_ref], dim=1)

        # normalize real_data
        real_data = real_data / 2.0
        ref_data = ref_data / 2.0

        assert -1 <= real_data.min() < 0
        assert 0 < real_data.max() <= 1

        assert -1 <= ref_data.min() < 0
        assert 0 < ref_data.max() <= 1

        real_image = real_data
        ref_image = ref_data

        D_real, real_label, pred_fake, fake_label, x_0_predict = self.forward(real_image, ref_image, coeff, pos_coeff,label)
        loss_D_real_GAN, loss_D_real_label = self.update_D(D_real, real_label, label)
        loss_D_fake_GAN, loss_D_fake_label = self.update_D(pred_fake, fake_label, mode=False)
        loss_D_GAN = loss_D_real_GAN + loss_D_fake_GAN
        loss_D_label = loss_D_real_label + loss_D_fake_label
        loss_D = loss_D_GAN + loss_D_label
        loss_G_GAN, loss_content, loss_G_label = self.update_G(pred_fake, x_0_predict, real_image, fake_label, label)
        loss_G = loss_G_GAN * self.opt.gamma_genL1 + loss_G_label * self.opt.gamma_genLabel + loss_content * self.opt.gamma_genMSE

        losses = {'g_loss_content': loss_content,
                  'g_loss_label': loss_G_label,
                  'g_loss_adv': loss_G_GAN,
                  'g_loss': loss_G,
                  'd_loss': loss_D,
                  'd_loss_adv': loss_D_GAN,
                  'd_loss_real_adv': loss_D_real_GAN,
                  'd_loss_fake_adv': loss_D_fake_GAN,
                  'd_loss_real_label': loss_D_real_label,
                  'd_loss_fake_label': loss_D_fake_label,
                  'd_loss_label': loss_D_label,
                  }
        if save_imag:
            x_t_1 = torch.randn_like(real_data)
            fake_sample = sample_from_model(
                pos_coeff, self.Gen, self.opt.num_timesteps, x_t_1, T, self.opt, ref_image, label)

            fake_sample *= 2

            if not self.opt.use_pytorch_wavelet:
                fake_sample = iwt(
                    fake_sample[:, :1], fake_sample[:, 1:2], fake_sample[:, 2:3], fake_sample[:, 3:4])
            else:
                fake_sample = iwt((fake_sample[:, :1], [torch.stack(
                    (fake_sample[:, 1:2], fake_sample[:, 2:3], fake_sample[:, 3:4]), dim=2)]))

            return losses,fake_label,fake_sample
        else:
            return losses,fake_label

    def test_(self, image, label, dwt, iwt, pos_coeff):
        T = get_time_schedule(self.opt, self.device)

        num_levels = int(np.log2(self.opt.ori_image_size // self.opt.current_resolution))
        # images = image
        realimage = image[:, 1, :, :].view([-1, 1, 256, 256])
        refimage = image[:, 0, :, :].view([-1, 1, 256, 256])
        if not self.opt.use_pytorch_wavelet:
            for i in range(num_levels):
                xll, xlh, xhl, xhh = dwt(realimage)
                xll_ref, xlh_ref, xhl_ref, xhh_ref = dwt(refimage)
        else:
            xll, xh = dwt(realimage)
            xll_ref, xh_ref = dwt(refimage)
            xlh, xhl, xhh = torch.unbind(xh[0], dim=2)
            xlh_ref, xhl_ref, xhh_ref = torch.unbind(xh_ref[0], dim=2)

        real_data = torch.cat([xll, xlh, xhl, xhh], dim=1)
        ref_data = torch.cat([xll_ref, xlh_ref, xhl_ref, xhh_ref], dim=1)

        # normalize real_data
        real_data = real_data / 2.0  # [-1, 1]
        ref_data = ref_data / 2.0

        assert -1 <= real_data.min() < 0
        assert 0 < real_data.max() <= 1

        assert -1 <= ref_data.min() < 0
        assert 0 < ref_data.max() <= 1
        ref_image = ref_data

        x_t_1 = torch.randn_like(real_data)
        fake_sample = sample_from_model(
        pos_coeff, self.Gen, self.opt.num_timesteps, x_t_1, T, self.opt, ref_image, label)

        fake_sample *= 2

        if not self.opt.use_pytorch_wavelet:
            fake_sample = iwt(
                fake_sample[:, :1], fake_sample[:, 1:2], fake_sample[:, 2:3], fake_sample[:, 3:4])
        else:
            fake_sample = iwt((fake_sample[:, :1], [torch.stack(
                (fake_sample[:, 1:2], fake_sample[:, 2:3], fake_sample[:, 3:4]), dim=2)]))

        return fake_sample

    def test_generate(self, image, label, dwt, iwt, pos_coeff):
        T = get_time_schedule(self.opt, self.device)

        num_levels = int(np.log2(self.opt.ori_image_size // self.opt.current_resolution))

        realimage = image[1, :, :].unsqueeze(0).unsqueeze(0).cuda()
        refimage = image[(0, 2), :, :].unsqueeze(0).cuda()

        label = label.cuda()
        if not self.opt.use_pytorch_wavelet:
            for i in range(num_levels):
                xll, xlh, xhl, xhh = dwt(realimage)
                xll_ref, xlh_ref, xhl_ref, xhh_ref = dwt(refimage)
        else:
            xll, xh = dwt(realimage)
            xll_ref, xh_ref = dwt(refimage)
            xlh, xhl, xhh = torch.unbind(xh[0], dim=2)
            xlh_ref, xhl_ref, xhh_ref = torch.unbind(xh_ref[0], dim=2)

        real_data = torch.cat([xll, xlh, xhl, xhh], dim=1)
        ref_data = torch.cat([xll_ref, xlh_ref, xhl_ref, xhh_ref], dim=1)

        # normalize real_data
        real_data = real_data / 2.0  # [-1, 1]
        ref_data = ref_data / 2.0

        assert -1 <= real_data.min() < 0
        assert 0 < real_data.max() <= 1

        assert -1 <= ref_data.min() < 0
        assert 0 < ref_data.max() <= 1
        ref_image = ref_data

        x_t_1 = torch.randn_like(real_data)
        fake_sample = sample_from_model(
            pos_coeff, self.Gen, self.opt.num_timesteps, x_t_1, T, self.opt, ref_image, label)

        fake_sample *= 2

        if not self.opt.use_pytorch_wavelet:
            fake_sample = iwt(
                fake_sample[:, :1], fake_sample[:, 1:2], fake_sample[:, 2:3], fake_sample[:, 3:4])
        else:
            fake_sample = iwt((fake_sample[:, :1], [torch.stack(
                (fake_sample[:, 1:2], fake_sample[:, 2:3], fake_sample[:, 3:4]), dim=2)]))
        #
        fake_image_rs = torch.cat((refimage[:, 0, :, :].view(-1, 1, 256, 256), \
                                   fake_sample, refimage[:, 1, :, :].view(-1, 1, 256, 256)), dim=1)
        return fake_image_rs.permute(0, 2, 3, 1).cpu().detach().numpy()


