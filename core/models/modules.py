#!/usr/bin/env python
import sys
import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
import numpy as np
import cupy as cp
import math

from core.models.links import CategoricalConditionalBatchNormalization
from core.models.layers import conv_Layer,dw_conv_Layer,deconv_Layer


class GlobalResBlock(chainer.Chain):
    def __init__(self, in_ch, out_ch, n_joint=10, g_div=10, norm='batch', activation=F.leaky_relu, ksize=3, stride=1, pad=1, downsample=False, upsample=False, residual=True, padding_mode=None):
        super(GlobalResBlock, self).__init__(
            c_l = conv_Layer(in_ch, in_ch, ksize=(ksize,1), stride=(stride,1), pad=(pad,0), normalize=norm, padding_mode=padding_mode),
            c_g = conv_Layer(in_ch, n_joint*in_ch//g_div, ksize=(ksize ,n_joint), stride = (stride,1), pad=(pad,0), normalize=norm, padding_mode=padding_mode),
            c_conc = conv_Layer(in_ch+in_ch//g_div, out_ch, ksize=(ksize, 1), stride=(stride,1), pad=(pad,0), normalize=norm, padding_mode=padding_mode)
        )
        self.activation = activation
        self.norm = norm
        self.n_joint = n_joint
        self.downsample = downsample
        self.upsample = upsample
        self.residual = residual
        self.learnable_sc = in_ch != out_ch and residual # False
        with self.init_scope():
            if self.learnable_sc:
                self.c_sc = conv_Layer(in_ch, out_ch, ksize=(1,1), stride=(1,1), pad=(0,0), normalize=norm, padding_mode=padding_mode)

    def __call__(self, x, y=None, outsize=None):
        bs, ch, t, wi = x.shape
        # source
        h = x
        if self.upsample:
            h = F.unpooling_2d(h, (2, 1), outsize=(t*2, wi)) if outsize is None else F.unpooling_2d(h, (2, 1), outsize=outsize)
        h_l = self.activation(self.c_l(h, y=y)) # y=None
        h_g = self.activation(self.c_g(h, y=y)) # y=None
        h_g = F.reshape(h_g, (bs, -1, h_g.shape[2], self.n_joint))
        h = self.c_conc(F.concat((h_l, h_g), axis=1), y=y)  # do not activate this
        if self.downsample:
            h = F.average_pooling_2d(h, ksize=(2, 1))

        # residual
        if self.residual:
            res = self.c_sc(x, y=y) if self.learnable_sc else x # x
            if self.upsample:
                res = F.unpooling_2d(res, (2, 1), outsize=(t*2, wi)) if outsize is None else F.unpooling_2d(h, (2, 1), outsize=outsize)
            if self.downsample:
                res = F.average_pooling_2d(res, ksize=(2, 1))

        return h + res if self.residual else h


class AttensionGate(chainer.Chain):
    def __init__(self, ch):
        super(AttensionGate, self).__init__(
            c_f = conv_Layer(ch, ch, ksize=(1,1), stride=(1,1), pad=(0,0), normalize=None),
            c_g = conv_Layer(ch, ch, ksize=(1,1), stride=(1,1), pad=(0,0), normalize=None),
            c_h = conv_Layer(ch, ch, ksize=(1,1), stride=(1,1), pad=(0,0), normalize=None),
        )

    def __call__(self, x):
        bs, ch, t, wi = x.shape
        f = F.reshape(F.leaky_relu(self.c_f(x)), (-1, t, wi))
        g = F.reshape(F.leaky_relu(self.c_g(x)), (-1, t, wi))
        m = F.softmax(F.batch_matmul(F.transpose(f, (0,2,1,)),g),axis=1)
        h = F.batch_matmul(m, F.reshape(F.leaky_relu(self.c_h(x)), (-1, t, wi)))
        h = F.reshape(h, (bs, ch, t, wi))
        return h

class AttensionResBlock(chainer.Chain):
    def __init__(self, in_ch, out_ch, norm=None, activation=F.leaky_relu, ksize=(3,3), stride=(1,1), pad=(1,1)):
        super(AttensionResBlock, self).__init__(
            c1 = conv_Layer(in_ch, out_ch, ksize=ksize, stride=stride, pad=pad, normalize=norm),
            c2 = conv_Layer(out_ch, out_ch, ksize=ksize, stride=stride, pad=pad, normalize=norm),
            atn = AttensionGate(out_ch)
        )
        self.activation = activation
        self.learnable_sc = in_ch != out_ch
        with self.init_scope():
            if self.learnable_sc:
                self.c_sc = conv_Layer(in_ch, out_ch, ksize=(1,1), stride=(1,1), pad=(0,0), normalize=norm)

    def __call__(self, x, gamma):
        h = x
        h = F.relu(self.c1(h))
        h = F.relu(self.c2(h))
        res = self.c_sc(x) if self.learnable_sc else x
        h = res + gamma * self.atn(h)
        return h

