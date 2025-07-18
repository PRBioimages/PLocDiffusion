# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from a file in the Score SDE library
# which was released under the Apache License.
#
# Source:
# https://github.com/yang-song/score_sde_pytorch/blob/main/models/layerspp.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_Apache). The modifications
# to this file are subject to the same Apache License.
# ---------------------------------------------------------------

# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
''' Codes adapted from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ncsnpp.py
'''

import functools

import numpy as np
import torch
import torch.nn as nn
from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
from einops import rearrange
from fp16_util import convert_module_to_f16, convert_module_to_f32
from . import dense_layer, layers, layerspp, utils, up_or_down_sampling


ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp_Adagn
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp_Adagn
ResnetBlockBigGAN_one = layerspp.ResnetBlockBigGANpp_Adagn_one
WaveletResnetBlockDDPM = layerspp.WaveletResnetBlockDDPMpp_Adagn
WaveletResnetBlockBigGAN = layerspp.WaveletResnetBlockBigGANpp_Adagn

Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
conv2d = dense_layer.conv2d
get_act = layers.get_act
default_initializer = layers.default_init
dense = dense_layer.dense
get_sinusoidal_positional_embedding = layers.get_timestep_embedding
ngf=8
####################################################################
#------------------------- Generator --------------------------
####################################################################
class label_gated_connection(nn.Module):
    def __init__(self, config, size):
        super().__init__()
        self.gate_fc = nn.Sequential(nn.Linear(config.num_class, int(size) * int(size)), nn.Sigmoid())
    def forward(self, y, skip_input, size):
        prob_map = self.gate_fc(y)
        prob_map = torch.reshape(prob_map, [-1, 1, size, size])
        output = torch.mul(skip_input, prob_map)
        return output


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


@utils.register_model(name='ncsnpp')
class NCSNpp(nn.Module):
    """NCSN++ model"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.not_use_tanh = config.not_use_tanh
        self.act = act = nn.SiLU()
        self.z_emb_dim = z_emb_dim = config.z_emb_dim

        self.patch_size = config.patch_size
        assert config.image_size % self.patch_size == 0

        self.nf = nf = config.num_channels_dae
        ch_mult = config.ch_mult
        self.num_res_blocks = num_res_blocks = config.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.attn_resolutions
        dropout = config.dropout
        resamp_with_conv = config.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [
            (config.image_size // self.patch_size) // (2 ** i) for i in range(num_resolutions)]

        self.conditional = conditional = config.conditional
        fir = config.fir
        fir_kernel = config.fir_kernel
        self.skip_rescale = skip_rescale = config.skip_rescale
        self.resblock_type = resblock_type = config.resblock_type.lower()
        self.progressive = progressive = config.progressive.lower()
        self.progressive_input = progressive_input = config.progressive_input.lower()
        self.embedding_type = embedding_type = config.embedding_type.lower()
        init_scale = 0.
        assert progressive in ['none', 'output_skip', 'residual']
        assert progressive_input in ['none', 'input_skip', 'residual']
        assert embedding_type in ['fourier', 'positional']
        combine_method = config.progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        modules = []
        modules_ref = []
        if embedding_type == 'fourier':

            modules.append(layerspp.GaussianFourierProjection(
                embedding_size=nf, scale=config.fourier_scale
            ))
            embed_dim = 2 * nf

        elif embedding_type == 'positional':
            embed_dim = nf
        else:
            raise ValueError(f'embedding type {embedding_type} unknown.')

        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
        AttnBlock = functools.partial(layerspp.AttnBlockpp,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale)

        Upsample = functools.partial(layerspp.Upsample,
                                     with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive == 'output_skip':
            self.pyramid_upsample = layerspp.Upsample(
                fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive == 'residual':
            pyramid_upsample = functools.partial(layerspp.Upsample,
                                                 fir=fir, fir_kernel=fir_kernel, with_conv=True)
        Downsample = functools.partial(layerspp.Downsample,
                                       with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive_input == 'input_skip':
            self.pyramid_downsample = layerspp.Downsample(
                fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive_input == 'residual':
            pyramid_downsample = functools.partial(layerspp.Downsample,
                                                   fir=fir, fir_kernel=fir_kernel, with_conv=True)

        if resblock_type == 'ddpm':
            ResnetBlock = functools.partial(ResnetBlockDDPM,
                                            act=act,
                                            dropout=dropout,
                                            init_scale=init_scale,
                                            skip_rescale=skip_rescale,
                                            temb_dim=nf * 4,
                                            zemb_dim=z_emb_dim)

        elif resblock_type == 'biggan':
            ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                            act=act,
                                            dropout=dropout,
                                            fir=fir,
                                            fir_kernel=fir_kernel,
                                            init_scale=init_scale,
                                            skip_rescale=skip_rescale,
                                            temb_dim=nf * 4,
                                            zemb_dim=z_emb_dim)
        elif resblock_type == 'biggan_oneadagn':
            ResnetBlock = functools.partial(ResnetBlockBigGAN_one,
                                            act=act,
                                            dropout=dropout,
                                            fir=fir,
                                            fir_kernel=fir_kernel,
                                            init_scale=init_scale,
                                            skip_rescale=skip_rescale,
                                            temb_dim=nf * 4,
                                            zemb_dim=z_emb_dim)

        else:
            raise ValueError(f'resblock type {resblock_type} unrecognized.')


        channels = config.num_channels * self.patch_size**2
        channels_ref = config.num_channels_ref * self.patch_size ** 2
        if progressive_input != 'none':
            input_pyramid_ch = channels

        modules.append(conv3x3(channels, nf))
        modules_ref.append(conv3x3(channels_ref, nf))
        hs_c = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                modules_ref.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                    modules_ref.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == 'ddpm':
                    modules.append(Downsample(in_ch=in_ch))
                    modules_ref.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))
                    modules_ref.append(ResnetBlock(down=True, in_ch=in_ch))
                if progressive_input == 'input_skip':
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == 'cat':
                        in_ch *= 2

                elif progressive_input == 'residual':
                    modules.append(pyramid_downsample(
                        in_ch=input_pyramid_ch, out_ch=in_ch))
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)


        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules_ref.append(ResnetBlock(in_ch=in_ch))
        modules.append(dense(in_ch * ngf * ngf, config.hiddenz_size))
        modules_ref.append(dense(in_ch * ngf * ngf, config.hiddenr_size))
        self.fc_ = dense(config.hiddenz_size + config.hiddenr_size + config.num_class, in_ch * ngf * ngf)
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))
        pyramid_ch = 0
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                ref_c = hs_c.pop()
                modules_ref.append(conv1x1(ref_c, ref_c))
                modules.append(label_gated_connection(config, nf / (2 ** (i_level-1))))
                modules.append(ResnetBlock(in_ch=in_ch + ref_c,
                                           out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if progressive != 'none':
                if i_level == num_resolutions - 1:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(
                            conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name.')
                else:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(
                            conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(pyramid_upsample(
                            in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name')

            if i_level != 0:
                if resblock_type == 'ddpm':
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if progressive != 'output_skip':
            channels = getattr(config, "num_out_channels", channels)
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)
        self.all_modules_ref = nn.ModuleList(modules_ref)

        mapping_layers = [PixelNorm(),
                          dense(config.nz, z_emb_dim),
                          self.act, ]
        for _ in range(config.n_mlp):
            mapping_layers.append(dense(z_emb_dim, z_emb_dim))
            mapping_layers.append(self.act)
        self.z_transform = nn.Sequential(*mapping_layers)

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.all_modules.apply(convert_module_to_f16)
        self.all_modules_ref.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.all_modules.apply(convert_module_to_f32)
        self.all_modules_ref.apply(convert_module_to_f32)

    def forward(self, x,ref, time_cond, z,yp):
        x = rearrange(x, "n c (h p1) (w p2) -> n (p1 p2 c) h w",
                      p1=self.patch_size, p2=self.patch_size)
        zemb = self.z_transform(z)
        modules = self.all_modules
        modules_ref = self.all_modules_ref
        m_idx = 0
        m_idx_ref = 0
        if self.embedding_type == 'fourier':
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1

        elif self.embedding_type == 'positional':
            timesteps = time_cond

            temb = layers.get_timestep_embedding(timesteps, self.nf)

        else:
            raise ValueError(f'embedding type {self.embedding_type} unknown.')

        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if not self.config.centered:
            x = 2 * x - 1.

        input_pyramid = None
        if self.progressive_input != 'none':
            input_pyramid = x

        hs = [modules[m_idx](x)]
        hs_ref = [modules_ref[m_idx_ref](ref)]
        m_idx_ref += 1
        m_idx += 1
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb, zemb)
                h_ref = modules_ref[m_idx_ref](hs_ref[-1], temb, zemb)
                m_idx += 1
                m_idx_ref += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    h_ref = modules_ref[m_idx_ref](h_ref)
                    m_idx += 1
                    m_idx_ref += 1

                hs.append(h)
                hs_ref.append(h_ref)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](hs[-1])
                    h_ref = modules_ref[m_idx_ref](hs_ref[-1])
                    m_idx += 1
                    m_idx_ref += 1
                else:
                    h = modules[m_idx](hs[-1], temb, zemb)
                    h_ref = modules_ref[m_idx_ref](hs_ref[-1], temb, zemb)
                    m_idx += 1
                    m_idx_ref += 1

                if self.progressive_input == 'input_skip':
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1

                elif self.progressive_input == 'residual':
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid

                hs.append(h)
                hs_ref.append(h_ref)

        h = hs[-1]
        h_ref = hs_ref[-1]
        h = modules[m_idx](h, temb, zemb)
        h_ref = modules_ref[m_idx_ref](h_ref, temb, zemb)
        m_idx += 1
        m_idx_ref += 1
        h = modules[m_idx](h.contiguous().view(-1, 512 * ngf * ngf))
        h_ref = modules_ref[m_idx_ref](h_ref.contiguous().view(-1, 512 * ngf * ngf))
        m_idx += 1
        m_idx_ref += 1
        h = torch.cat((h, h_ref, yp), dim=1)
        h = self.act(self.fc_(h).view([-1, 512, ngf, ngf]))
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb, zemb)
        m_idx += 1

        pyramid = None

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h_enc = hs.pop()
                href_enc = hs_ref.pop()
                href_enc = href_enc * self.act(modules_ref[m_idx_ref](href_enc))
                m_idx_ref += 1
                skip_enc = modules[m_idx](yp, h_enc + href_enc, h_enc.shape[-1])
                m_idx += 1
                h = modules[m_idx](torch.cat([h, skip_enc], dim=1), temb, zemb)
                m_idx += 1

            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if self.progressive != 'none':
                if i_level == self.num_resolutions - 1:
                    if self.progressive == 'output_skip':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    elif self.progressive == 'residual':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(
                            f'{self.progressive} is not a valid name.')
                else:
                    if self.progressive == 'output_skip':
                        pyramid = self.pyramid_upsample(pyramid)
                        pyramid_h = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == 'residual':
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(
                            f'{self.progressive} is not a valid name')

            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb, zemb)
                    m_idx += 1

        assert not hs

        if self.progressive == 'output_skip':
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        assert m_idx == len(modules)
        # unpatchify
        h = rearrange(h, "n (c p1 p2) h w -> n c (h p1) (w p2)",
                      p1=self.patch_size, p2=self.patch_size)

        if not self.not_use_tanh:
            return torch.tanh(h)
        else:
            return h


@utils.register_model(name='wavelet_ncsnpp')
class WaveletNCSNpp(NCSNpp):
    """NCSN++ model"""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.not_use_tanh = config.not_use_tanh
        self.act = act = nn.SiLU()
        self.z_emb_dim = z_emb_dim = config.z_emb_dim

        self.patch_size = config.patch_size
        assert config.image_size % self.patch_size == 0

        self.nf = nf = config.num_channels_dae
        ch_mult = config.ch_mult
        self.num_res_blocks = num_res_blocks = config.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.attn_resolutions
        dropout = config.dropout
        resamp_with_conv = config.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [
            (config.image_size // self.patch_size) // (2 ** i) for i in range(num_resolutions)]

        self.conditional = conditional = config.conditional
        fir = config.fir
        fir_kernel = config.fir_kernel
        self.skip_rescale = skip_rescale = config.skip_rescale
        self.resblock_type = resblock_type = config.resblock_type.lower()
        self.progressive = progressive = config.progressive.lower()
        self.progressive_input = progressive_input = config.progressive_input.lower()
        self.embedding_type = embedding_type = config.embedding_type.lower()
        init_scale = 0.
        assert progressive in ['none', 'output_skip', 'residual']
        assert progressive_input in ['none', 'input_skip', 'residual']
        assert embedding_type in ['fourier', 'positional']
        combine_method = config.progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        self.no_use_fbn = getattr(self.config, "no_use_fbn", False)
        self.no_use_freq = getattr(self.config, "no_use_freq", False)
        self.no_use_residual = getattr(self.config, "no_use_residual", False)

        modules = []
        modules_ref = []

        if embedding_type == 'fourier':

            modules.append(layerspp.GaussianFourierProjection(
                embedding_size=nf, scale=config.fourier_scale
            ))
            embed_dim = 2 * nf

        elif embedding_type == 'positional':
            embed_dim = nf

        else:
            raise ValueError(f'embedding type {embedding_type} unknown.')

        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        AttnBlock = functools.partial(layerspp.AttnBlockpp,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale)

        Upsample = functools.partial(layerspp.Upsample,
                                     with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive == 'output_skip':
            self.pyramid_upsample = layerspp.Upsample(
                fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive == 'residual':
            pyramid_upsample = functools.partial(layerspp.Upsample,
                                                 fir=fir, fir_kernel=fir_kernel, with_conv=True)

        Downsample = functools.partial(layerspp.Downsample,
                                       with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive_input == 'input_skip':
            self.pyramid_downsample = layerspp.Downsample(
                fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive_input == 'residual':
            if self.no_use_residual:
                pyramid_downsample = functools.partial(layerspp.Downsample,
                                                       fir=fir, fir_kernel=fir_kernel, with_conv=True)
            else:
                pyramid_downsample = functools.partial(
                    layerspp.WaveletDownsample)

        if resblock_type == 'ddpm':
            ResnetBlock = functools.partial(ResnetBlockDDPM,
                                            act=act,
                                            dropout=dropout,
                                            init_scale=init_scale,
                                            skip_rescale=skip_rescale,
                                            temb_dim=nf * 4,
                                            zemb_dim=z_emb_dim)

        elif resblock_type == 'biggan':
            if self.no_use_freq:
                ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                                act=act,
                                                dropout=dropout,
                                                fir=fir,
                                                fir_kernel=fir_kernel,
                                                init_scale=init_scale,
                                                skip_rescale=skip_rescale,
                                                temb_dim=nf * 4,
                                                zemb_dim=z_emb_dim)
            else:
                ResnetBlock = functools.partial(WaveletResnetBlockBigGAN,
                                                act=act,
                                                dropout=dropout,
                                                init_scale=init_scale,
                                                skip_rescale=skip_rescale,
                                                temb_dim=nf * 4,
                                                zemb_dim=z_emb_dim)

        elif resblock_type == 'biggan_oneadagn':
            ResnetBlock = functools.partial(ResnetBlockBigGAN_one,
                                            act=act,
                                            dropout=dropout,
                                            fir=fir,
                                            fir_kernel=fir_kernel,
                                            init_scale=init_scale,
                                            skip_rescale=skip_rescale,
                                            temb_dim=nf * 4,
                                            zemb_dim=z_emb_dim)

        else:
            raise ValueError(f'resblock type {resblock_type} unrecognized.')

        # Downsampling block

        channels = config.num_channels * self.patch_size**2
        channels_ref = config.num_channels_ref * self.patch_size ** 2
        if progressive_input != 'none':
            input_pyramid_ch = channels

        modules.append(conv3x3(channels, nf))
        modules_ref.append(conv3x3(channels_ref, nf))

        hs_c = [nf]
        hs_c2 = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                modules_ref.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                    modules_ref.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                hs_c2.append(in_ch)
                if resblock_type == 'ddpm':
                    modules.append(Downsample(in_ch=in_ch))
                    modules_ref.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))
                    modules_ref.append(ResnetBlock(down=True, in_ch=in_ch))

                if progressive_input == 'input_skip':
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == 'cat':
                        in_ch *= 2

                elif progressive_input == 'residual':
                    modules.append(pyramid_downsample(
                        in_ch=input_pyramid_ch, out_ch=in_ch))
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules_ref.append(ResnetBlock(in_ch=in_ch))
        modules.append(dense(in_ch*8*8,config.hiddenz_size))
        modules_ref.append(dense(in_ch*8*8,config.hiddenr_size))
        self.fc_ = dense(config.hiddenz_size+config.hiddenr_size+config.num_class,in_ch*8*8)
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))
        pyramid_ch = 0
        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                ref_c = hs_c.pop()
                modules_ref.append(conv1x1(ref_c, ref_c))
                modules.append(label_gated_connection(config, nf/(2**i_level)))
                modules.append(ResnetBlock(in_ch=in_ch + ref_c,
                                           out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if progressive != 'none':
                if i_level == num_resolutions - 1:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(
                            conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name.')
                else:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(
                            conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(pyramid_upsample(
                            in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name')

            if i_level != 0:
                if resblock_type == 'ddpm':
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    if self.no_use_freq:
                        modules.append(ResnetBlock(in_ch=in_ch, up=True))
                    else:
                        modules.append(ResnetBlock(
                            in_ch=in_ch, up=True, hi_in_ch=hs_c2.pop()))

        assert not hs_c

        if progressive != 'output_skip':
            channels = getattr(config, "num_out_channels", channels)
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)
        self.all_modules_ref = nn.ModuleList(modules_ref)
        mapping_layers = [PixelNorm(),
                          dense(config.nz, z_emb_dim),
                          self.act, ]
        for _ in range(config.n_mlp):
            mapping_layers.append(dense(z_emb_dim, z_emb_dim))
            mapping_layers.append(self.act)
        self.z_transform = nn.Sequential(*mapping_layers)

        self.dwt = DWT_2D("haar")
        self.iwt = IDWT_2D("haar")

    def forward(self, x, ref, time_cond, z, yp):
        import time
        x = rearrange(x, "n c (h p1) (w p2) -> n (p1 p2 c) h w",
                      p1=self.patch_size, p2=self.patch_size)
        zemb = self.z_transform(z)
        modules = self.all_modules
        modules_ref = self.all_modules_ref
        m_idx = 0
        m_idx_ref = 0
        if self.embedding_type == 'fourier':
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1
        elif self.embedding_type == 'positional':
            timesteps = time_cond

            temb = layers.get_timestep_embedding(timesteps, self.nf)

        else:
            raise ValueError(f'embedding type {self.embedding_type} unknown.')

        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if not self.config.centered:
            x = 2 * x - 1.
            ref = 2*ref-1.

        # Downsampling block
        input_pyramid = None
        if self.progressive_input != 'none':
            input_pyramid = x

        hs = [modules[m_idx](x)]
        hs_ref = [modules_ref[m_idx_ref](ref)]

        skipHs = []
        skipHs_ref = []
        m_idx += 1
        m_idx_ref += 1
        end = time.time()
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb, zemb)
                h_ref = modules_ref[m_idx_ref](hs_ref[-1], temb, zemb)
                m_idx += 1
                m_idx_ref += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    h_ref = modules_ref[m_idx_ref](h_ref)
                    m_idx += 1
                    m_idx_ref += 1

                hs.append(h)
                hs_ref.append(h_ref)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](h)
                    h_ref = modules_ref[m_idx_ref](h_ref)
                    m_idx += 1
                    m_idx_ref += 1
                else:
                    if self.no_use_freq:
                        h = modules[m_idx](h, temb, zemb)
                        h_ref = modules_ref[m_idx_ref](h_ref, temb, zemb)
                    else:
                        h, skipH = modules[m_idx](h, temb, zemb)
                        h_ref, skipH_ref = modules_ref[m_idx_ref](h_ref, temb, zemb)
                        skipHs.append(skipH)
                        skipHs_ref.append(skipH_ref)
                    m_idx += 1
                    m_idx_ref += 1

                if self.progressive_input == 'input_skip':
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1

                elif self.progressive_input == 'residual':
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid

                hs.append(h)
                hs_ref.append(h_ref)


        h = hs[-1]
        h_ref = hs_ref[-1]
        if self.no_use_fbn:
            h = modules[m_idx](h, temb, zemb)
            h_ref = modules_ref[m_idx_ref](h_ref, temb, zemb)
        else:
            h, hlh, hhl, hhh = self.dwt(h)
            h = modules[m_idx](h / 2., temb, zemb)
            h = self.iwt(h * 2., hlh, hhl, hhh)

            h_ref , hlh_ref , hhl_ref , hhh_ref  = self.dwt(h_ref )
            h_ref  = modules_ref[m_idx_ref ](h_ref  / 2., temb, zemb)
            h_ref = self.iwt(h_ref  * 2., hlh_ref , hhl_ref , hhh_ref)

        m_idx += 1
        m_idx_ref += 1
        h = modules[m_idx](h.reshape(-1,512*8*8))
        h_ref = modules_ref[m_idx_ref](h_ref.reshape(-1,512*8*8))
        m_idx += 1
        m_idx_ref += 1
        h = torch.cat((h, h_ref, yp), dim=1)
        h = self.act(self.fc_(h).reshape([-1,512,8,8]))
        # attn block
        h = modules[m_idx](h)
        m_idx += 1

        if self.no_use_fbn:
            h = modules[m_idx](h, temb, zemb)
        else:
            # forward on original feature space
            h = modules[m_idx](h, temb, zemb)
            h, hlh, hhl, hhh = self.dwt(h)
            h = modules[m_idx](h / 2., temb, zemb)
            h = self.iwt(h * 2., hlh, hhl, hhh)
        m_idx += 1

        pyramid = None
        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h_enc = hs.pop()
                href_enc = hs_ref.pop()
                href_enc = href_enc*self.act(modules_ref[m_idx_ref](href_enc))

                m_idx_ref += 1
                skip_enc = modules[m_idx](yp,h_enc+href_enc,h_enc.shape[-1])
                m_idx += 1
                h = modules[m_idx](torch.cat([h, skip_enc], dim=1), temb, zemb)

                m_idx += 1

            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if self.progressive != 'none':
                if i_level == self.num_resolutions - 1:
                    if self.progressive == 'output_skip':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    elif self.progressive == 'residual':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(
                            f'{self.progressive} is not a valid name.')
                else:
                    if self.progressive == 'output_skip':
                        pyramid = self.pyramid_upsample(pyramid)
                        pyramid_h = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == 'residual':
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(
                            f'{self.progressive} is not a valid name')

            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    if self.no_use_freq:
                        h = modules[m_idx](h, temb, zemb)
                    else:
                        h = modules[m_idx](h, temb, zemb, skipH=skipHs.pop())

                    m_idx += 1



        assert not hs
        if self.progressive == 'output_skip':
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        assert m_idx == len(modules)
        # unpatchify
        h = rearrange(h, "n (c p1 p2) h w -> n c (h p1) (w p2)",
                      p1=self.patch_size, p2=self.patch_size)

        if not self.not_use_tanh:
            return torch.tanh(h)
        else:
            return h



####################################################################
#------------------------- Discriminators --------------------------
####################################################################

class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, act=nn.LeakyReLU(0.2)):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            dense(embedding_dim, hidden_dim),
            act,
            dense(hidden_dim, output_dim),
        )

    def forward(self, temp):
        temb = get_sinusoidal_positional_embedding(temp, self.embedding_dim)
        temb = self.main(temb)
        return temb


class DownConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        t_emb_dim=128,
        downsample=False,
        act=nn.LeakyReLU(0.2),
        fir_kernel=(1, 3, 3, 1)
    ):
        super().__init__()

        self.fir_kernel = fir_kernel
        self.downsample = downsample

        self.conv1 = nn.Sequential(
            conv2d(in_channel, out_channel, kernel_size, padding=padding),
        )

        self.conv2 = nn.Sequential(
            conv2d(out_channel, out_channel, kernel_size, padding=padding, init_scale=0.)
        )
        self.dense_t1 = dense(t_emb_dim, out_channel)

        self.act = act

        self.skip = nn.Sequential(
            conv2d(in_channel, out_channel, 1, padding=0, bias=False),
        )

    def forward(self, input, t_emb):
        out = self.act(input)
        out = self.conv1(out)
        out += self.dense_t1(t_emb)[..., None, None]
        out = self.act(out)

        if self.downsample:
            out = up_or_down_sampling.downsample_2d(out, self.fir_kernel, factor=2)
            input = up_or_down_sampling.downsample_2d(input, self.fir_kernel, factor=2)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / np.sqrt(2)

        return out

class Discriminator(nn.Module):
    """A time-dependent discriminator for large images (CelebA, LSUN)."""

    def __init__(self, nc=1, ngf=32, t_emb_dim=128, act=nn.LeakyReLU(0.2), patch_size=1, use_local_loss=False, num_layers=6,use_cond=True):
        super().__init__()
        self.patch_size = patch_size
        self.use_local_loss = use_local_loss

        nc = nc * self.patch_size * self.patch_size
        # Gaussian random feature embedding layer for time
        self.act = act

        self.t_embed = TimestepEmbedding(
            embedding_dim=t_emb_dim,
            hidden_dim=t_emb_dim,
            output_dim=t_emb_dim,
            act=act,
        )

        self.start_conv = conv2d(nc, ngf * 2, 1, padding=0)
        self.conv1 = DownConvBlock(ngf * 2, ngf * 4, t_emb_dim=t_emb_dim, downsample=True, act=act)

        self.conv2 = DownConvBlock(ngf * 4, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)

        self.conv3 = DownConvBlock(ngf * 8, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)

        self.conv4, self.conv5, self.conv6 = None, None, None
        if num_layers >= 4:
            self.conv4 = DownConvBlock(ngf * 8, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)
        if num_layers >= 5:
            self.conv5 = DownConvBlock(ngf * 8, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)
        if num_layers >= 6:
            self.conv6 = DownConvBlock(ngf * 8, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)

        self.final_conv = conv2d(ngf * 8 + 1, ngf * 8, 3, padding=1)
        self.end_linear = dense(ngf * 8, 1)
        if use_local_loss:
            self.local_end_linear = dense(ngf * 8, 1)
        if use_cond:
            self.cond_linear = dense(ngf*8,2)
            self.softmax = nn.Softmax(dim=1)

        self.stddev_group = 5
        self.stddev_feat = 1

    def forward(self, x, t, x_t,use_cond=True):
        x = rearrange(x, "n c (h p1) (w p2) -> n (c p1 p2) h w", p1=self.patch_size, p2=self.patch_size)
        x_t = rearrange(x_t, "n c (h p1) (w p2) -> n (c p1 p2) h w", p1=self.patch_size, p2=self.patch_size)
        t_embed = self.act(self.t_embed(t))

        input_x = torch.cat((x, x_t), dim=1)

        h = self.start_conv(input_x)
        h = self.conv1(h, t_embed)

        h = self.conv2(h, t_embed)

        h = self.conv3(h, t_embed)
        if self.conv4 is not None:
            h = self.conv4(h, t_embed)
        if self.conv5 is not None:
            h = self.conv5(h, t_embed)
        if self.conv6 is not None:
            h = self.conv6(h, t_embed)
        out = h

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)

        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        out = self.act(out)

        out = out.view(out.shape[0], out.shape[1], -1)
        t = out
        out = self.end_linear(out.sum(2))
        if self.use_local_loss:
            out2 = self.local_end_linear(t.permute(0, 2, 1))
            return (out, out2)
        if use_cond:
            cond_pred = self.softmax(self.cond_linear(t.sum(2)))
            return out,cond_pred

        return out