#!/usr/bin/env python
import sys
import math

import numpy as np
import cupy as cp
import chainer
from chainer.backends import cuda
from chainer.functions.array.broadcast import broadcast_to
from chainer.functions.connection import embed_id, linear, convolution_2d, deconvolution_2d, depthwise_convolution_2d
from chainer.links.connection.linear import Linear
from chainer.initializers import normal
import chainer.functions as F
import chainer.links as L
from chainer import Variable

from .function import max_singular_value
from core.models.links import CategoricalConditionalBatchNormalization, InstanceNormalization, ConditionalInstanceNormalization

class conv_Layer(chainer.Chain):
    def __init__(self, ch_in, ch_out, ksize=(3,1), stride=(1,1), pad=(1, 0), dilate=(1,1), groups=1, wscale=0.02, normalize=None, use_wscale=False, n_class=2, padding_mode=None):
        assert normalize in ['batch', 'cbatch', 'spectrum+batch', 'instance', 'conditional_instance', None]
        # wの初期値
        if use_wscale:
            w = chainer.initializers.Normal(1.0) 
        else:
            if type(ksize) == 'int':
                w = chainer.initializers.Uniform(scale=math.sqrt(1/(ch_in/groups)/ksize/ksize))
            else:
                w = chainer.initializers.Uniform(scale=math.sqrt(1/(ch_in/groups)/ksize[0]/ksize[1]))

        super(conv_Layer, self).__init__()
        with self.init_scope():
            # conv層
            pad_conv = 0 if padding_mode else pad # pad
            if normalize == 'spectrum' or normalize == 'spectrum+batch':
                self.conv = SNConvolution2D(ch_in, ch_out, ksize=ksize, stride=stride, pad=pad_conv, dilate=dilate, groups=groups, initialW=w)
            else:
                self.conv = L.Convolution2D(ch_in, ch_out, ksize=ksize, stride=stride, pad=pad_conv, dilate=dilate, groups=groups, initialW=w)

            # normalize
            if normalize == 'batch' or normalize == 'spectrum+batch':
                self.batch_norm = L.BatchNormalization(ch_out)
            elif normalize == 'cbatch':
                self.batch_norm = CategoricalConditionalBatchNormalization(ch_out, n_class)
            elif normalize == 'instance':
                self.batch_norm = InstanceNormalization(ch_out)
            elif normalize == 'conditional_instance':
                #self.batch_norm = ConditionalInstanceNormalization(ch_out)
                self.norm0 = InstanceNormalization(ch_out)    
                self.norm1 = InstanceNormalization(ch_out)    
 
        self.c = np.sqrt(2 / ((ch_in / groups) * ksize[0] * ksize[1]))
        self.normalize = normalize # 'batch'
        self.use_wscale = use_wscale # None
        self.padding_mode = padding_mode # None
        self.padding_width = [[0,0],[0,0],[pad[0],pad[0]],[pad[1],pad[1]]] # None

    def __call__(self, x, y=None):
        h = x * self.c if self.use_wscale else x

        if self.padding_mode:
            h = F.pad(h, self.padding_width, self.padding_mode)

        h = self.conv(h)

        if self.normalize == 'batch' or self.normalize == 'spectrum+batch':
            h = self.batch_norm(h)
        elif self.normalize == 'cbatch':
            h = self.batch_norm(h, y)
        elif self.normalize == 'instance':
            h = self.batch_norm(h)
        elif self.normalize == 'conditional_instance':
            #h = self.batch_norm(h, y)
            with cuda.get_device_from_id(self._device_id):
                one = self.xp.ones(y.shape, dtype=h.dtype)
            h = self.norm0(h) * y + self.norm1(h) * (one-y)

        return h

class dw_conv_Layer(chainer.Chain):
    def __init__(self, ch_in, ch_out, ksize=(3,1), stride=(1,1), pad=(1,0),wscale=0.02, normalize=None, use_wscale=False, n_class=2, padding_mode=None):
        assert normalize in ['batch', 'cbatch', 'spectrum+batch', 'instance', 'conditional_instance', None]
        if use_wscale:
            w = chainer.initializers.Normal(1.0) 
        else:
            if type(ksize) == 'int':
                w = chainer.initializers.Uniform(scale=math.sqrt(1/ksize/ksize))
            else:
                w = chainer.initializers.Uniform(scale=math.sqrt(1/ksize[0]/ksize[1]))
        super(dw_conv_Layer, self).__init__()
        pad_conv = 0 if padding_mode else pad
        with self.init_scope():
            if normalize == 'spectrum' or normalize == 'spectrum+batch':
                self.conv = SNDepthwiseConvolution2D(ch_in, ch_out//ch_in, ksize=ksize, stride=stride, pad=pad_conv, initialW=w)
            else: 
                self.conv = L.DepthwiseConvolution2D(ch_in, ch_out//ch_in, ksize=ksize, stride=stride, pad=pad_conv, initialW=w)
                
            if normalize == 'batch' or normalize == 'spectrum+batch':
                self.batch_norm = L.BatchNormalization(ch_out)
            elif normalize == 'cbatch':
                self.batch_norm = CategoricalConditionalBatchNormalization(ch_out, n_class)
            elif normalize == 'instance':
                #self.batch_norm = L.BatchNormalization(axis=(0,3,4)) 
                self.batch_norm = InstanceNormalization(ch_out)
            elif normalize == 'conditional_instance':
                #self.batch_norm = ConditionalInstanceNormalization(ch_out)
                self.norm0 = InstanceNormalization(ch_out)    
                self.norm1 = InstanceNormalization(ch_out)    
        
        self.c = np.sqrt(2 / (ksize[0] * ksize[1]))
        self.normalize = normalize
        self.use_wscale = use_wscale
        self.padding_mode = padding_mode
        self.padding_width = [[0,0],[0,0],[pad[0],pad[0]],[pad[1],pad[1]]]

    def __call__(self, x, y=None):
        h = x * self.c if self.use_wscale else x
        if self.padding_mode:
            h = F.pad(h, self.padding_width, mode=self.padding_mode)
        h = self.conv(h)
        if self.normalize == 'batch' or self.normalize == 'spectrum+batch':
            h = self.batch_norm(h)
        elif self.normalize == 'cbatch':
            h = self.batch_norm(h, y)
        elif self.normalize == 'instance':
            h = self.batch_norm(h)
        elif self.normalize == 'conditional_instance':
            with cuda.get_device_from_id(self._device_id):
                one = self.xp.ones(y.shape, dtype=h.dtype)
            h = self.norm0(h) * y + self.norm1(h) * (one-y)
        return h

class deconv_Layer(chainer.Chain):
    def __init__(self, ch_in, ch_out, ksize=(3,1), stride=(1,1), pad=(2,0), wscale=0.02, normalize=None, use_wscale=False, n_class=2):
        assert normalize in ['batch', 'cbatch', 'spectrum+batch', 'instance', 'conditional_instance', None]
        # wの初期値
        if use_wscale:
            w = chainer.initializers.Normal(1.0) 
        else:
            if type(ksize) == 'int':
                w = chainer.initializers.Uniform(scale=math.sqrt(1/ch_in/ksize/ksize))
            else:
                w = chainer.initializers.Uniform(scale=math.sqrt(1/ch_in/ksize[0]/ksize[1]))

        super(deconv_Layer, self).__init__()
        with self.init_scope():
            # deconv
            if normalize == 'spectrum' or normalize == 'spectrum+batch':
                self.deconv = SNDeconvolution2D(ch_in, ch_out, ksize=ksize, stride=stride, pad=pad, initialW=w)
            else:
                self.deconv = L.Deconvolution2D(ch_in, ch_out, ksize=ksize, stride=stride, pad=pad, initialW=w)

            # normalize
            if normalize == 'batch' or normalize == 'spectrum+batch':
                self.batch_norm = L.BatchNormalization(ch_out)
            elif normalize == 'cbatch':
                self.batch_norm = CategoricalConditionalBatchNormalization(ch_out, n_class)
            elif normalize == 'instance':
                self.batch_norm = InstanceNormalization(ch_out)
            elif normalize == 'conditional_instance':
                #self.batch_norm = ConditionalInstanceNormalization(ch_out)
                self.norm0 = InstanceNormalization(ch_out)    
                self.norm1 = InstanceNormalization(ch_out)    
  
        self.c = np.sqrt(2 / (ch_in * ksize[0] * ksize[1]))
        self.normalize = normalize # 'batch'
        self.use_wscale = use_wscale # None

    def __call__(self, x, y=None):
        h = x * self.c if self.use_wscale else x

        h = self.deconv(h)

        if self.normalize == 'batch' or self.normalize == 'spectrum+batch':
            h = self.batch_norm(h)  
        elif self.normalize == 'cbatch':
            h = self.batch_norm(h, y)
        elif self.normalize == 'instance':
            h = self.batch_norm(h)
        elif self.normalize == 'conditional_instance':
            #h = self.batch_norm(h, y)
            with cuda.get_device_from_id(self._device_id):
                one = self.xp.ones(y.shape, dtype=h.dtype)
            h = self.norm0(h) * y + self.norm1(h) * (one-y)
            
        return h


class linear_Layer(chainer.Chain):
    def __init__(self, ch_in, ch_out, use_wscale=False):
        if use_wscale:
            w = chainer.initializers.Normal(1.0)
        else:
            w = chainer.initializers.Uniform(scale=math.sqrt(1/ch_in))
        self.c = np.sqrt(2.0/ch_in)
        self.use_wscale = use_wscale
        super(linear_Layer, self).__init__()
        with self.init_scope():
            self.l = L.Linear(ch_in, ch_out, initialW=w)
    def __call__(self, x):
        h = x * self.c if self.use_wscale else x
        return self.l(h)


class SNConvolution2D(L.Convolution2D):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, dilate=1, groups=1, nobias=False, initialW=None, initial_bias=None, use_gamma=False, Ip=1, factor=None):
        self.Ip = Ip
        self.use_gamma = use_gamma
        self.factor = factor
        super(SNConvolution2D, self).__init__(
            in_channels, out_channels, ksize, stride, pad, nobias, initialW, initial_bias, dilate=dilate, groups=groups)
        self.u = np.random.normal(size=(1, out_channels)).astype(dtype="f")
        self.register_persistent('u')

    @property
    def W_bar(self):
        #Spectral Normalized Weight
        W_mat = self.W.reshape(self.W.shape[0], -1)
        sigma, _u, _ = max_singular_value(W_mat, self.u, self.Ip)
        if self.factor:
            sigma = sigma / self.factor
        sigma = broadcast_to(sigma.reshape((1,1,1,1)), self.W.shape)
        if chainer.config.train:
            self.u[:] = _u
        if hasattr(self, 'gamma'):
            return broadcast_to(self.gamma, self.W.shape) * self.W / sigma
        else:
            return self.W / sigma

    def _initialize_params(self, in_size):
        super(SNConvolution2D, self)._initialize_params(in_size)
        if self.use_gamma:
            W_mat = self.W.data.reshape(self.W.shape[0], -1)
            _, s, _ = np.linalg.svd(W_mat)
            with self.init_scope():
                self.gamma = chainer.Parameter(s[0], (1,1,1,1))

    def __call__(self, x):
        if self.W.data is None:
            self._initialize_params(x.shape[1])
        return convolution_2d.convolution_2d(x, self.W_bar, self.b, self.stride, self.pad, dilate=self.dilate, groups=self.groups)

class SNDepthwiseConvolution2D(L.DepthwiseConvolution2D):
    def __init__(self, in_channels, channel_multiplier, ksize, stride=1, pad=0, nobias=False, initialW=None, initial_bias=None, use_gamma=False, Ip=1, factor=None):
        self.Ip = Ip
        self.use_gamma = use_gamma
        self.factor = factor
        super(SNDepthwiseConvolution2D, self).__init__(
            in_channels, channel_multiplier, ksize, stride, pad, nobias, initialW, initial_bias)
        self.u = np.random.normal(size=(1, channel_multiplier)).astype(dtype="f")
        self.register_persistent('u')

    @property
    def W_bar(self):
        #Spectral Normalized Weight
        W_mat = self.W.reshape(self.W.shape[0], -1)
        sigma, _u, _ = max_singular_value(W_mat, self.u, self.Ip)
        if self.factor:
            sigma = sigma / self.factor
        sigma = broadcast_to(sigma.reshape((1,1,1,1)), self.W.shape)
        if chainer.config.train:
            self.u[:] = _u
        if hasattr(self, 'gamma'):
            return broadcast_to(self.gamma, self.W.shape) * self.W / sigma
        else:
            return self.W / sigma

    def _initialize_params(self, in_size):
        super(SNDepthwiseConvolution2D, self)._initialize_params(in_size)
        if self.use_gamma:
            W_mat = self.W.data.reshape(self.W.shape[0], -1)
            _, s, _ = np.linalg.svd(W_mat)
            with self.init_scope():
                self.gamma = chainer.Parameter(s[0], (1, 1, 1, 1))

    def __call__(self, x):
        if self.W.data is None:
            self._initialize_params(x.shape[1])
        return depthwise_convolution_2d.depthwise_convolution_2d(x, self.W_bar, self.b, self.stride, self.pad)

class SNDeconvolution2D(L.Convolution2D):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, groups=1, nobias=False, initialW=None, initial_bias=None, use_gamma=False, Ip=1, factor=None):
        self.Ip = Ip
        self.use_gamma = use_gamma
        self.factor = factor
        super(SNDeconvolution2D, self).__init__(
            in_channels, out_channels, ksize, stride, pad, nobias, initialW, initial_bias, groups=groups)
        self.u = np.random.normal(size=(1, out_chanels)).astype(dtype="f")
        self.register_persistent('u')

    @property
    def W_bar(self):
        #Spectral Normalized Weight
        W_mat = self.W.reshape(self.W.shape[0], -1)
        sigma, _u, _ = max_singular_value(W_mat, self.u, self.Ip)
        if self.factor:
            sigma = sigma / self.factor
        sigma = broadcast_to(sigma.reshape((1,1,1,1)), self.W.shape)
        if chainer.config.train:
            self.u[:] = _u
        if hasattr(self, 'gamma'):
            return broadcast_to(self.gamma, self.W.shape) * self.W / sigma
        else:
            return self.W / sigma

    def _initialize_params(self, in_size):
        super(SNDeconvolution2D, self)._initialize_params(in_size)
        if self.use_gamma:
            W_mat = self.W.data.reshape(self.W.shape[0], -1)
            _, s, _ = np.linalg.svd(W_mat)
            with self.init_scope():
                self.gamma = chainer.Parameter(s[0], (1,1,1,1))

    def __call__(self, x):
        if self.W.data is None:
            self._initialize_params(x.shape[1])
        return deconvolution_2d.deconvolution_2d(x, self.W_bar, self.b, self.stride, self.pad, groups=self.groups)


class SNEmbedID(chainer.link.Link):

    ignore_label = None

    def __init__(self, in_size, out_size, initialW=None, ignore_label=None, Ip=1, factor=None):
        super(SNEmbedID, self).__init__()
        self.ignore_label = ignore_label
        self.Ip = Ip
        self.factor = factor
        with self.init_scope():
            if initialW is None:
                initialW = normal.Normal(1.0)
            self.W = variable.Parameter(initialW, (in_size, out_size))

        self.u = np.random.normal(size=(1, in_size)).astype(dtype="f")
        self.register_persistent('u')

    @property
    def W_bar(self):
        sigma, _u, _ = max_singular_value(self.W, self.u, self.Ip)
        if self.factor:
            sigma = sigma / self.factor
        sigma = broadcast_to(sigma.reshape((1, 1)), self.W.shape)
        self.u[:] = _u
        return self.W / sigma

    def __call__(self, x):
        return embed_id.embed_id(x, self.W_bar, ignore_label=self.ignore_label)    


class SNLinear(Linear):
    def __init__(self, in_size, out_size, use_gamma=False, nobias=False,
                 initialW=None, initial_bias=None, Ip=1, factor=None):
        self.Ip = Ip
        self.use_gamma = use_gamma
        self.factor = factor
        super(SNLinear, self).__init__(
            in_size, out_size, nobias, initialW, initial_bias
        )
        self.u = np.random.normal(size=(1, out_size)).astype(dtype="f")
        self.register_persistent('u')

    @property
    def W_bar(self):
        sigma, _u, _ = max_singular_value(self.W, self.u, self.Ip)
        if self.factor:
            sigma = sigma / self.factor
        sigma = broadcast_to(sigma.reshape((1, 1)), self.W.shape)
        self.u[:] = _u
        if hasattr(self, 'gamma'):
            return broadcast_to(self.gamma, self.W.shape) * self.W / sigma
        else:
            return self.W / sigma

    def _initialize_params(self, in_size):
        super(SNLinear, self)._initialize_params(in_size)
        if self.use_gamma:
            _, s, _ = np.linalg.svd(self.W.data)
            with self.init_scope():
                self.gamma = chainer.Parameter(s[0], (1, 1))

    def __call__(self, x):
        if self.W.data is None:
            self._initialize_params(x.size // x.shape[0])
        return linear.linear(x, self.W_bar, self.b)


