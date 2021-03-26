#!/usr/bin/env python
import os
import sys
import csv

import chainer
import chainer.functions as F
from chainer import Variable
import numpy as np
import cupy
import random
import subprocess

from core.models.loss_function import sigmoid_cross_entropy


def loss_ls_dis_real(dis):
    loss = (dis[:,0] - 1.0) ** 2
    return loss

def loss_ls_dis_fake(dis):
    loss = (dis[:,0] - 0.) ** 2
    return loss

def loss_ls_gen(dis):
    loss = (dis[:,0] - 1.0) ** 2
    return loss

def loss_hinge_dis_real(dis):
    loss = F.relu(1.0 - dis[:,0])
    return loss
    
def loss_hinge_dis_fake(dis):
    loss = F.relu(1.0 + dis[:,0])
    return loss

def loss_hinge_gen(dis):
    loss = - dis[:,0]
    return loss

def loss_class(dis, cls):
    if len(cls.shape) == 4:
        cls = cls[:,:,0,0]
    #loss = sigmoid_cross_entropy(dis[:,1:], cls)
    loss = sigmoid_cross_entropy(dis, cls)
    return loss

class StarGANUpdater(chainer.training.updaters.StandardUpdater):
    def __init__(self, *args, **kwargs):
        models = kwargs.pop('models')
        self.gen, self.dis = models
        self.cfg = kwargs.pop('cfg')
        self.save_dir = kwargs.pop('out')

        self.loss_type = self.cfg.train.loss_type if hasattr(self.cfg.train,'loss_type') else 'ls'
        if self.loss_type == 'wgan-gp':
            self._lam_d_gp = self.cfg.train.parameters.lam_d_gp
            self._lam_d_drift = self.cfg.train.parameters.lam_d_drift
        self._lam_g_ad = self.cfg.train.parameters.lam_g_ad
        self._lam_d_ad = self.cfg.train.parameters.lam_d_ad
        self._lam_g_rec = self.cfg.train.parameters.lam_g_rec
        self._lam_g_eq = self.cfg.train.parameters.lam_g_eq
        self._lam_g_style = self.cfg.train.parameters.lam_g_style
        self._lam_d_style = self.cfg.train.parameters.lam_d_style
        self._lam_g_cont = self.cfg.train.parameters.lam_g_cont
        self._lam_d_cont = self.cfg.train.parameters.lam_d_cont
        self._lam_g_sm = self.cfg.train.parameters.lam_g_sm

        self.criterion = self.cfg.train.criterion if hasattr(self.cfg.train, 'criterion') else 'l2'
        self._learning_rate_anneal = self.cfg.train.parameters.learning_rate_anneal
        self._learning_rate_anneal_interval = self.cfg.train.parameters.learning_rate_anneal_interval
        self.preview_interval = self.cfg.train.preview_interval

        self._iter = 0
        self.xp = self.gen.xp

        super(StarGANUpdater, self).__init__(*args, **kwargs)

    def update_core(self):
        self._iter += 1

        opt_gen = self.get_optimizer('gen')
        opt_dis = self.get_optimizer('dis')

        ## Create batch
        batch = self.get_iterator('main').next()
        batchsize = len(batch)

        # get data
        x_batch, x_class_labels, y_batch, y_class_labels, cont_id = [self.xp.expand_dims(self.xp.array(b), axis=1).astype("f") for b in zip(*batch)]
        x_data = self.converter(x_batch, self.device)
        y_data = self.converter(y_batch, self.device)
        x_labels = self.converter(x_class_labels.transpose(0,3,1,2), self.device)
        y_labels = self.converter(y_class_labels.transpose(0,3,1,2), self.device)
        cont_label = self.converter(cont_id.transpose(0,3,1,2), self.device)


        ## Forward
        x = Variable(x_data)
        y_data = Variable(y_data)
        x_labels = Variable(x_labels)
        y_labels = Variable(y_labels)

        x_y = self.gen(x, F.concat((x_labels, y_labels), axis=1))
        if self._lam_g_rec > 0:
            x_y_x = self.gen(x_y, F.concat((y_labels, x_labels), axis=1))

        ## Annealing learning rate
        if self._learning_rate_anneal > 0 and self._iter % self._learning_rate_anneal_interval == 0:
            opt_gen.alpha *= 1.0 - self._learning_rate_anneal
            opt_dis.alpha *= 1.0 - self._learning_rate_anneal


        #============================
        #
        ##  update Discriminator
        #
        #============================

        d_x_y = self.dis(x_y)
        d_y = self.dis(y_data)

        # mask for adv and equal loss (if x_class == y_class then 0 else 1)
        isDiffTargetAndInput = self.xp.sum(x_labels[:,:,0,0].data != y_labels[:,:,0,0].data, axis=1).astype(float) * 0.5 if self._lam_g_eq > 0 else self.xp.zeros(batchsize)

        # weight for adv and equal loss
        # w_adv have weight only for data which x_class != y_class
        w_adv = isDiffTargetAndInput * float(batchsize) / (self.xp.sum(isDiffTargetAndInput.astype(float)) + 1e-6)
        # w_eq have weight only for data which x_class = y_class
        w_eq =  (1.0 - isDiffTargetAndInput) * float(batchsize) / (self.xp.sum(1.0 - isDiffTargetAndInput.astype(float)) + 1e-6)

        # style-class loss and content-class loss
        loss_dis_style = F.average(loss_class(d_y[:,1:1+y_labels.shape[1]], y_labels))
        loss_dis_cont = F.average(loss_class(d_y[:,-cont_label.shape[1]:], cont_label))

        # adv loss 
        if self.loss_type == 'hinge':
            loss_dis_adv_fake = F.average(w_adv * loss_hinge_dis_fake(d_x_y))
            loss_dis_adv_real = F.average(w_adv * loss_hinge_dis_real(d_y))
            loss_dis_adv = loss_dis_adv_fake + loss_dis_adv_real
            loss_dis = self._lam_d_ad * loss_dis_adv + self._lam_d_style * loss_dis_style + self._lam_d_cont * loss_dis_cont

        elif self.loss_type == 'ls':        
            loss_dis_adv_fake = F.average(w_adv * loss_ls_dis_fake(d_x_y))
            loss_dis_adv_real = F.average(w_adv * loss_ls_dis_real(d_y))
            loss_dis_adv = loss_dis_adv_fake + loss_dis_adv_real
            loss_dis = self._lam_d_ad * loss_dis_adv + self._lam_d_style * loss_dis_style + self._lam_d_cont * loss_dis_cont

        elif self.loss_type == 'wgan-gp':
            loss_dis_adv = F.average(d_x_y[:,0] - d_y[:,0])
            # calcurate GP
            epsilon = self.xp.random.rand(batchsize,1,1,1).astype("f")
            y_hat = Variable(epsilon * x_y.data + (1-epsilon) * y_data .data)
            d_y_hat = self.dis(y_hat)
            g_d, = chainer.grad([w_adv * d_y_hat[:,:1]], [y_hat], enable_double_backprop=True)
            g_d_norm = F.sqrt(F.batch_l2_norm_squared(g_d) + 1e-6)
            loss_dis_gp = F.mean_squared_error(g_d_norm, self.xp.ones_like(g_d_norm.data))
            loss_dis_drift = F.average(d_y[:,0]*d_y[:,0])

            loss_dis = self._lam_d_ad * loss_dis_adv + self._lam_d_style * loss_dis_style + self._lam_d_cont * loss_dis_cont + self._lam_d_gp * loss_dis_gp + self._lam_d_drift * loss_dis_drift
            chainer.report({'loss_gp': self._lam_d_gp*loss_dis_gp}, self.dis)
            chainer.report({'loss_drift': self._lam_d_drift*loss_dis_drift}, self.dis)

        else:
            print(f'invalid loss type!!! ({self.loss_type})')
            assert False

        chainer.report({'loss_adv': self._lam_d_ad*loss_dis_adv}, self.dis)
        chainer.report({'loss_style': self._lam_d_style*loss_dis_style}, self.dis)
        chainer.report({'loss_cont': self._lam_d_cont*loss_dis_cont}, self.dis)
        self.dis.cleargrads()
        loss_dis.backward()
        opt_dis.update()


        #============================
        #
        ##  update Generator
        #
        #============================

        d_x_y2 = self.dis(x_y)

        # adv loss
        if self.loss_type == 'hinge':
            loss_gen_adv = F.average(w_adv * loss_hinge_gen(d_x_y2))
        elif self.loss_type == 'ls':
            loss_gen_adv = F.average(w_adv * loss_ls_gen(d_x_y2))
        elif self.loss_type == 'wgan-gp':
            loss_gen_adv = F.average(w_adv * -d_x_y2[:,0])
        else:
            print(f'invalid loss type!!! ({self.loss_type})')
            assert False

        # equal loss
        if self.criterion == 'l2':
            loss_gen_equal = F.average(w_eq * F.average(F.squared_error(x_y, x), axis=(1,2,3)))
        elif self.criterion == 'l1':
            loss_gen_equal = F.average(w_eq * F.average(F.absolute_error(x_y, x), axis=(1,2,3)))

        # smoothness loss
        loss_gen_sm = F.mean_absolute_error(x_y[:,:,1:,:], x_y[:,:,:-1,:])

        # style-class loss and content-class loss
        loss_gen_style = F.average(loss_class(d_x_y2[:,1:1+y_labels.shape[1]], y_labels))
        loss_gen_cont = F.average(loss_class(d_x_y2[:,-cont_label.shape[1]:], cont_label))
 
        # cyclic loss
        if self.criterion == 'l2':
            loss_rec = F.mean_squared_error(x_y_x, x) if self._lam_g_rec > 0 else 0
        elif self.criterion == 'l1':
            loss_rec = F.mean_absolute_error(x_y_x, x) if self._lam_g_rec > 0 else 0

        loss_gen = self._lam_g_ad * loss_gen_adv + self._lam_g_sm * loss_gen_sm + self._lam_g_style * loss_gen_style + self._lam_g_cont * loss_gen_cont + self._lam_g_rec * loss_rec
        if self.cfg.train.class_equal:
            loss_gen += self._lam_g_eq * loss_gen_equal

        if loss_dis_adv.data < 0.5 * loss_gen_adv.data:
            n_gen = 5
        else:
            n_gen = 1
        
        for _ in range(n_gen):
            self.gen.cleargrads()
            loss_gen.backward()
            opt_gen.update()

        chainer.report({'loss_rec': loss_rec * self._lam_g_rec},  self.gen)
        if self.cfg.train.class_equal:
            chainer.report({'loss_eq': loss_gen_equal * self._lam_g_eq},  self.gen)
        chainer.report({'loss_sm': loss_gen_sm * self._lam_g_sm},  self.gen)
        chainer.report({'loss_adv': loss_gen_adv * self._lam_g_ad}, self.gen)
        chainer.report({'loss_style': loss_gen_style * self._lam_g_style}, self.gen)
        chainer.report({'loss_cont': loss_gen_cont * self._lam_g_cont}, self.gen)
        chainer.report({'lr': opt_gen.alpha})

        # save preview
        if self._iter % self.preview_interval == 0:
            save_path = os.path.join(self.save_dir, 'preview')
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            x = chainer.backends.cuda.to_cpu(x.data)
            x_y = chainer.backends.cuda.to_cpu(x_y.data)
            x_y_x = chainer.backends.cuda.to_cpu(x_y_x.data)
            y_data = chainer.backends.cuda.to_cpu(y_data.data)
            data_list = np.concatenate([x, x_y, x_y_x, y_data], axis=1)

            np.save(os.path.join(save_path, f'iter_{self._iter:04d}'), data_list)

            # save x_class, y_class and cont_id of preview
            with open(os.path.join(save_path, 'preview.txt'), 'a') as f:
                source = np.where(chainer.backends.cuda.to_cpu(x_class_labels)==1)[-1]
                target = np.where(chainer.backends.cuda.to_cpu(y_class_labels)==1)[-1]
                cont = np.where(chainer.backends.cuda.to_cpu(cont_id)==1)[-1]
                f.write(f'iter {self._iter:04d}\n')
                for b in range(batchsize):
                    f.write(f'{b}: {source[b]} -> {target[b]} ({cont[b]})\n')
                f.write('\n')
