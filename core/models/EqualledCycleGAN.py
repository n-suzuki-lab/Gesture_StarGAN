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

## Generator model
class EqualledCycleGAN_Generator7(chainer.Chain):
    def __init__(self, cfg, n_class=2, activation=F.leaky_relu):
        top = cfg.top if hasattr(cfg, 'top') else 176
        g_div = cfg.g_div if hasattr(cfg, 'g_div') else 16
        norm = cfg.norm if hasattr(cfg, 'norm') else 'batch'
        padding_mode = cfg.padding_mode if hasattr(cfg, 'padding_mode') else None
        self.use_sigmoid = cfg.use_sigmoid if hasattr(cfg, 'use_sigmoid') else False

        self.activation = activation
        self.n_class = n_class
        self.n_joint = 10

        super(EqualledCycleGAN_Generator7, self).__init__()
        with self.init_scope():
            self.c_in = conv_Layer(n_class*2+1, top, ksize=(7,1), stride=(1,1), pad=(3,0), normalize=norm, padding_mode=padding_mode)
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
        style_label = F.broadcast_to(style_label, (bs, self.n_class*2, hi, wi))
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

## Alternative model of generator which has no local filters and only global filters.
class EqualledCycleGAN_Generator8(chainer.Chain):
    def __init__(self, cfg, n_class=2, n_joint=10, activation=F.leaky_relu):
        top = cfg.top if hasattr(cfg, 'top') else 32
        norm = cfg.norm if hasattr(cfg, 'norm') else 'conditional_instance'
        padding_mode = cfg.padding_mode if hasattr(cfg, 'padding_mode') else None
        ch_min = cfg.ch_min if hasattr(cfg, 'ch_min') else 1

        assert top % ch_min == 0
        self.activation = activation
        self.n_class = n_class
        self.n_joint = 10
        self.ch_min = ch_min

        super(EqualledCycleGAN_Generator8, self).__init__()
        with self.init_scope():
            self.c_in = conv_Layer(n_class*2+1, top, ksize=(7,self.n_joint*3), stride=(1,1), pad=(3,0), normalize=norm, padding_mode=padding_mode)
            self.db1 = conv_Layer(ch_min, top*2, ksize=(7,top//ch_min), stride=(2,1), pad=(3,0), normalize=norm, padding_mode=padding_mode)
            self.db2 = conv_Layer(ch_min, top*4, ksize=(7,top*2//ch_min), stride=(2,1), pad=(3,0), normalize=norm, padding_mode=padding_mode)

            self.ub1 = conv_Layer(ch_min, top*2, ksize=(7,top*4//ch_min), stride=(1,1), pad=(3,0), normalize=norm)
            self.ub2 = conv_Layer(ch_min, top, ksize=(7,top*2//ch_min), stride=(1,1), pad=(3,0), normalize=norm)
            self.c_out = conv_Layer(ch_min, self.n_joint*3, ksize=(7,top//ch_min), stride=(1,1), pad=(3,0), normalize=None)

    def __call__(self, x, style_label=None):
        assert style_label is not None
        h = x
        bs, ch, hi, wi = h.shape
        style_label = F.broadcast_to(style_label, (bs, self.n_class*2, hi, wi))
        h = F.concat((h, style_label), axis=1)

        h = self.activation(self.c_in(h))
        h = F.reshape(h, (bs, self.ch_min, hi, -1))

        #encode
        for i in range(1,3):
            h = getattr(self, 'db'+str(i))(h)
            h = F.reshape(h, (bs, self.ch_min, h.shape[2], -1))
        #decode
        for i in range(1,3):
            h = getattr(self, 'ub'+str(i))(h)
            h = F.reshape(h, (bs, self.ch_min, h.shape[2], -1))
            h = F.unpooling_2d(h, (2, 1), outsize=(h.shape[2]*2,h.shape[3]))
 
        h = getattr(self,'c_out')(h)
        h = F.reshape(h, (bs, 1, hi, -1))
        return h


## Discriminator model
class EqualledCycleGAN_ArgumentedDiscriminator(chainer.Chain):
    def __init__(self, cfg, n_class=4, n_joint=10, n_gesture=4):
        top = cfg.top if hasattr(cfg, 'top') else 32
        norm = cfg.norm if hasattr(cfg, 'norm') else 'batch'
        padding_mode = cfg.padding_mode if hasattr(cfg, 'padding_mode') else None
        ch_min = cfg.ch_min if hasattr(cfg, 'ch_min') else 1

        assert top % ch_min == 0

        self.noise= cfg.noise if hasattr(cfg, 'noise') else False
        self.dropout = cfg.dropout if hasattr(cfg, 'dropout') else False

        self.ch_min = ch_min
        self.n_joint = n_joint
        self.n_class = n_class
        self.n_gesture = n_gesture
        

        super(EqualledCycleGAN_ArgumentedDiscriminator, self).__init__()
        with self.init_scope():
            self.conv1 = conv_Layer(1, top, ksize=(7,3), stride=(1,1), pad=(3,1), normalize=norm, padding_mode=padding_mode)
            self.conv2 = conv_Layer(top, top*2, ksize=(7,4), stride=(1,1), pad=(3,0), normalize=norm, padding_mode=padding_mode)
            self.fc1 = L.Linear(None, 500)
            self.fc2 = L.Linear(None, 1+n_class+n_gesture)

    def __call__(self, x, test=False):
        h = x
        h = F.max_pooling_2d(F.leaky_relu(self.conv1(h)), ksize=2)
        h = F.max_pooling_2d(F.leaky_relu(self.conv2(h)), ksize=2)
        h = self.fc1(h)
        h = self.fc2(F.leaky_relu(h))
        return h


class EqualledCycleGAN_ArgumentedDiscriminator1(chainer.Chain):
    def __init__(self, cfg, n_class=2, n_joint=10, n_gesture=4):
        top = cfg.top if hasattr(cfg, 'top') else 176
        norm = cfg.norm if hasattr(cfg, 'norm') else 'batch'
        padding_mode = cfg.padding_mode if hasattr(cfg, 'padding_mode') else None
        g_div = cfg.g_div if hasattr(cfg, 'g_div') else 10        

        self.noise= cfg.noise if hasattr(cfg, 'noise') else False
        self.dropout = cfg.dropout if hasattr(cfg, 'dropout') else False

        self.n_joint = n_joint
        self.n_class = n_class
        

        super(EqualledCycleGAN_ArgumentedDiscriminator, self).__init__()
        with self.init_scope():
            self.c_in = conv_Layer(1, top, ksize=(7,1), stride=(1,1), pad=(3,0), normalize=norm, padding_mode=padding_mode)
            self.db1 = GlobalResBlock(top, top, g_div=g_div, ksize=3, norm=norm, downsample=True, residual=False, padding_mode=padding_mode)
            self.db2 = GlobalResBlock(top, top, g_div=g_div, ksize=3, norm=norm, downsample=True, residual=False, padding_mode=padding_mode)
            self.db3 = GlobalResBlock(top, top, g_div=g_div, ksize=3, norm=norm, downsample=True, residual=False, padding_mode=padding_mode)
            self.last = L.Linear(None, 1+n_class+n_gesture)
         

    def __call__(self, x, test=False):
        h = x
        h = F.leaky_relu(self.c_in(h))
        for i in range(3):
            h = getattr(self, 'db'+str(i+1))(h)
            if self.noise:
                h = add_noise(h)
            if self.dropout:
                h = F.dropout(h)
            h = F.leaky_relu(h)
        h = self.last(h)
        return h

## Alternative model of discriminator which has no local filters and only global filters.
class EqualledCycleGAN_ArgumentedDiscriminator2(chainer.Chain):
    def __init__(self, cfg, n_class=2, n_joint=10, n_gesture=4):
        top = cfg.top if hasattr(cfg, 'top') else 32
        norm = cfg.norm if hasattr(cfg, 'norm') else 'batch'
        padding_mode = cfg.padding_mode if hasattr(cfg, 'padding_mode') else None
        ch_min = cfg.ch_min if hasattr(cfg, 'ch_min') else 1

        assert top % ch_min == 0

        self.noise= cfg.noise if hasattr(cfg, 'noise') else False
        self.dropout = cfg.dropout if hasattr(cfg, 'dropout') else False

        self.ch_min = ch_min
        self.n_joint = n_joint
        self.n_class = n_class
        

        super(EqualledCycleGAN_ArgumentedDiscriminator2, self).__init__()
        with self.init_scope():
            self.c_in = conv_Layer(1, top, ksize=(7, self.n_joint*1), stride=(1,1), pad=(3,0), normalize=norm, padding_mode=padding_mode)
            self.db1 = conv_Layer(ch_min, top*2, ksize=(5, top//ch_min), stride=(2,1), pad=(2,0), normalize=norm, padding_mode=padding_mode)
            self.db2 = conv_Layer(ch_min, top*4, ksize=(5, top*2//ch_min), stride=(2,1), pad=(2,0), normalize=norm, padding_mode=padding_mode)
            self.db3 = conv_Layer(ch_min, top*8, ksize=(5, top*4//ch_min), stride=(2,1), pad=(2,0), normalize=norm, padding_mode=padding_mode)
            self.last = L.Linear(None, 1+n_class+n_gesture)
         

    def __call__(self, x, test=False):
        h = x
        bs, _, hi, wi = h.shape
        h = F.leaky_relu(self.c_in(h))
        h = F.reshape(h, (bs, self.ch_min, h.shape[2], -1))
        for i in range(3):
            h = getattr(self, 'db'+str(i+1))(h)
            if self.noise:
                h = add_noise(h)
            if self.dropout:
                h = F.dropout(h)
            h = F.leaky_relu(h)
            h = F.reshape(h, (bs, self.ch_min, h.shape[2], -1))
        h = self.last(h)
        return h


## Patch GAN like discriminator
class EqualledCycleGAN_ArgumentedPatchDiscriminator(chainer.Chain):
    def __init__(self, cfg, n_joint=10):
        top = cfg.top if hasattr(cfg, 'top') else 176
        norm = cfg.norm if hasattr(cfg, 'norm') else 'batch'
        padding_mode = cfg.padding_mode if hasattr(cfg, 'padding_mode') else None
        
        self.noise= cfg.noise if hasattr(cfg, 'noise') else False
        self.dropout = cfg.dropout if hasattr(cfg, 'dropout') else False

        self.n_joint = 10

        super(EqualledCycleGAN_ArgumentedPatchDiscriminator, self).__init__()
        with self.init_scope():
            self.c_in = conv_Layer(1, top, ksize=(7,3), stride=(1,3), pad=(3,0), normalize=norm, padding_mode=padding_mode)
            self.db1 = GlobalResBlock(top, top, g_div=16, ksize=3, norm=norm, downsample=True, residual=False, padding_mode=padding_mode)
            self.db2 = GlobalResBlock(top, top, g_div=16, ksize=3, norm=norm, downsample=True, residual=False, padding_mode=padding_mode)
            self.db3 = GlobalResBlock(top, top, g_div=16, ksize=3, norm=norm, downsample=True, residual=False, padding_mode=padding_mode)
            self.c_last = conv_Layer(top, 2, ksize=(3,3), stride=(1,1), pad=(1,1), normalize=None, padding_mode=padding_mode)


    def __call__(self, x, test=False):
        h = x
        h = F.leaky_relu(self.c_in(h))
        for i in range(3):
            h = getattr(self, 'db'+str(i+1))(h)
            if self.noise:
                h = add_noise(h)
            if self.dropout:
                h = F.dropout(h)
            h = F.leaky_relu(h)
        h = self.c_last(h)
        return h

class EqualledCycleGAN_ArgumentedDiscriminator3(chainer.Chain):
    def __init__(self, cfg, n_class=4, n_joint=10, n_user=4):
        # top = cfg.top if hasattr(cfg, 'top') else 32
        # norm = cfg.norm if hasattr(cfg, 'norm') else 'batch'
        # padding_mode = cfg.padding_mode if hasattr(cfg, 'padding_mode') else None
        # ch_min = cfg.ch_min if hasattr(cfg, 'ch_min') else 1

        # assert top % ch_min == 0

        # self.noise= cfg.noise if hasattr(cfg, 'noise') else False
        # self.dropout = cfg.dropout if hasattr(cfg, 'dropout') else False

        # self.ch_min = ch_min
        self.n_joint = n_joint
        self.n_class = n_class
        self.n_user = n_user

        super(EqualledCycleGAN_ArgumentedDiscriminator3, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 32, ksize=(11,3))
            self.conv2 = L.Convolution2D(None, 64, ksize=(6,2))
            self.fc1 = L.Linear(None, 500)
            self.fc2 = L.Linear(None, 1+n_class+n_user)
         

    def __call__(self, x, test=False):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=(3,2))
        h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=3)
        h = self.fc1(h)
        h = self.fc2(F.relu(h))
        return h

# without gesture recognition
class EqualledCycleGAN_ArgumentedDiscriminator4(chainer.Chain):
    def __init__(self, cfg, n_class=2, n_joint=10, n_gesture=4):
        # top = cfg.top if hasattr(cfg, 'top') else 32
        # norm = cfg.norm if hasattr(cfg, 'norm') else 'batch'
        # padding_mode = cfg.padding_mode if hasattr(cfg, 'padding_mode') else None
        # ch_min = cfg.ch_min if hasattr(cfg, 'ch_min') else 1

        # assert top % ch_min == 0

        # self.noise= cfg.noise if hasattr(cfg, 'noise') else False
        # self.dropout = cfg.dropout if hasattr(cfg, 'dropout') else False

        # self.ch_min = ch_min
        self.n_joint = n_joint
        self.n_class = n_class
        self.n_gesture = n_gesture

        super(EqualledCycleGAN_ArgumentedDiscriminator4, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 32, ksize=(11,3))
            self.conv2 = L.Convolution2D(None, 64, ksize=(6,2))
            self.fc1 = L.Linear(None, 500)
            self.fc2 = L.Linear(None, 1+n_class)
         

    def __call__(self, x, test=False):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=(3,2))
        h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=3)
        h = self.fc1(h)
        h = self.fc2(F.relu(h))
        return h



