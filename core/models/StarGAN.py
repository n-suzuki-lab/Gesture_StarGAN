#!/usr/bin/env python
import sys
import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
from chainer import Variable
import numpy as np
import cupy as cp
import math

from core.models.layers import conv_Layer, dw_conv_Layer, deconv_Layer
from core.models.modules import AttensionGate, AttensionResBlock, GlobalResBlock


#=======================================
#
#   Generator Model
#
#=======================================

class StarGAN_Generator(chainer.Chain):
    def __init__(self, cfg, n_style=2, activation=F.leaky_relu):
        top = cfg.top if hasattr(cfg, 'top') else 176
        g_div = cfg.g_div if hasattr(cfg, 'g_div') else 16
        norm = cfg.norm if hasattr(cfg, 'norm') else 'batch'
        padding_mode = cfg.padding_mode if hasattr(cfg, 'padding_mode') else None
        self.use_sigmoid = cfg.use_sigmoid if hasattr(cfg, 'use_sigmoid') else False

        self.activation = activation
        self.n_style = n_style

        super(StarGAN_Generator, self).__init__()
        with self.init_scope():
            self.c_in = conv_Layer(n_style*2+1, top, ksize=(7,1), stride=(1,1), pad=(3,0), normalize=norm, padding_mode=padding_mode)
            self.db1 = GlobalResBlock(top, top*2, g_div=10, ksize=3, norm=norm, downsample=True, residual=False, padding_mode=padding_mode)
            self.db2 = GlobalResBlock(top*2, top*4, g_div=10, ksize=3, norm=norm, downsample=True, residual=False, padding_mode=padding_mode)

            self.rb1 = GlobalResBlock(top*4, top*4, g_div=10, ksize=3, norm=norm, padding_mode=padding_mode)
            self.rb2 = GlobalResBlock(top*4, top*4, g_div=10, ksize=3, norm=norm, padding_mode=padding_mode)
            self.rb3 = GlobalResBlock(top*4, top*4, g_div=10, ksize=3, norm=norm, padding_mode=padding_mode)

            self.ub1 = GlobalResBlock(top*4, top*2, g_div=10, ksize=3, norm=norm, upsample=True, residual=False, padding_mode=padding_mode)
            self.ub2 = GlobalResBlock(top*2, top, g_div=10, ksize=3, norm=norm, upsample=True, residual=False, padding_mode=padding_mode)
            self.c_out = deconv_Layer(top, 1, ksize=(7,1), stride=(1,1), pad=(3,0), normalize=None)

    def __call__(self, x, style_label=None, volatile=False):
        assert style_label is not None
        h = x
        bs, ch, hi, wi = h.shape
        style_label = F.broadcast_to(style_label, (bs, self.n_style*2, hi, wi))
        h = F.concat((h, style_label), axis=1)

        h = self.activation(self.c_in(h))

        #encode
        for i in range(1,3):
            h = getattr(self, 'db'+str(i))(h)
        #resblock
        for i in range(1,4):
            h = getattr(self, 'rb'+str(i))(h)
        #decode
        for i in range(1,3):
            h = getattr(self, 'ub'+str(i))(h)
 
        h = getattr(self,'c_out')(h)
        if self.use_sigmoid:
            h = F.sigmoid(h)
        return h


#=======================================
#
#   Discriminator Model
#
#=======================================


class StarGAN_Discriminator(chainer.Chain):
    def __init__(self, cfg, n_gesture=4, n_user=4):
        super(StarGAN_Discriminator, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 32, ksize=(11,3))
            self.conv2 = L.Convolution2D(None, 64, ksize=(6,2))
            self.fc1 = L.Linear(None, 500)
            self.fc2 = L.Linear(None, 1+n_gesture+n_user)
         

    def __call__(self, x, test=False):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=(3,2))
        h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=3)
        h = self.fc1(h)
        h = self.fc2(F.relu(h))
        return h



