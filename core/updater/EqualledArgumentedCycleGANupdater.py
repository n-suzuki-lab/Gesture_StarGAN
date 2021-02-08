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

from core.utils.XYZnormalize import Normalize
#from core.utils.make_gif_fromposition import show_pose_multi
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

class EqualledArgumentedCycleGANUpdater(chainer.training.updaters.StandardUpdater):
    def __init__(self, *args, **kwargs):
        models = kwargs.pop('models')
        self.gen, self.dis = models
        self.cfg = kwargs.pop('cfg')
        # self.datamax = kwargs.pop('datamax')
        # self.datamin = kwargs.pop('datamin')
        self.save_dir = kwargs.pop('out')

        self.loss_type = self.cfg.train.loss_type if hasattr(self.cfg.train,'loss_type') else 'ls'
        if self.loss_type == 'wgan-gp':
            self._lam_d_gp = self.cfg.train.parameters.lam_d_gp
            self._lam_d_drift = self.cfg.train.parameters.lam_d_drift
        self._lam_g_ad = self.cfg.train.parameters.lam_g_ad
        self._lam_d_ad = self.cfg.train.parameters.lam_d_ad
        self._lam_g_rec = self.cfg.train.parameters.lam_g_rec
        self._lam_g_eq = self.cfg.train.parameters.lam_g_eq
        self._lam_g_ges = self.cfg.train.parameters.lam_g_ges
        self._lam_d_ges = self.cfg.train.parameters.lam_d_ges
        self._lam_g_user = self.cfg.train.parameters.lam_g_user
        self._lam_d_user = self.cfg.train.parameters.lam_d_user
        self._lam_g_sm = self.cfg.train.parameters.lam_g_sm

        self.criterion = self.cfg.train.criterion if hasattr(self.cfg.train, 'criterion') else 'l2'
        self._learning_rate_anneal = self.cfg.train.parameters.learning_rate_anneal
        self._learning_rate_anneal_interval = self.cfg.train.parameters.learning_rate_anneal_interval
        self.preview_interval = self.cfg.train.preview_interval

        self._iter = 0
        self.xp = self.gen.xp

        super(EqualledArgumentedCycleGANUpdater, self).__init__(*args, **kwargs)

    def update_core(self):
        self._iter += 1

        opt_gen = self.get_optimizer('gen')
        opt_dis = self.get_optimizer('dis')

        ## Create batch
        batch = self.get_iterator('main').next()
        batchsize = len(batch)

        # get data
        x_batch, labels, y_batch, target_labels, user_id = [self.xp.expand_dims(self.xp.array(b), axis=1).astype("f") for b in zip(*batch)]
        x_data = self.converter(x_batch, self.device)
        x_labels_data = self.converter(labels.transpose(0,3,1,2), self.device)
        x_target_labels_data = self.converter(target_labels.transpose(0,3,1,2), self.device)
        y_real_data = self.converter(y_batch, self.device)
        user_label = self.converter(user_id.transpose(0,3,1,2), self.device)


        ## Forward
        x = Variable(x_data) # 変換前データ
        y_real = Variable(y_real_data) # 変換後realデータ
        x_labels = Variable(x_labels_data) # 変換前ラベル
        x_target_labels = Variable(x_target_labels_data) # 変換後ラベル

        x_y = self.gen(x, F.concat((x_labels, x_target_labels),axis=1)) # fakeデータ
        if self._lam_g_rec > 0:
            x_y_x = self.gen(x_y, F.concat((x_target_labels, x_labels),axis=1)) # fakeデータ

        ## Annealing learning rate
        if self._learning_rate_anneal > 0 and self._iter % self._learning_rate_anneal_interval == 0:
            opt_gen.alpha *= 1.0 - self._learning_rate_anneal
            opt_dis.alpha *= 1.0 - self._learning_rate_anneal

        ## update Discriminator
        d_x_y = self.dis(x_y)
        d_real = self.dis(y_real)

        #変換元と変換先のラベルが異なる場合1、同じ場合0
        isDiffTargetAndInput = self.xp.sum(x_labels[:,:,0,0].data != x_target_labels[:,:,0,0].data, axis=1).astype(float) * 0.5 if self._lam_g_eq > 0 else self.xp.zeros(batchsize)

        # class loss
        # loss_dis_cls = F.average(loss_class(d_real[:,1:], x_target_labels)) # リアルデータを元に学習
        loss_dis_ges = F.average(loss_class(d_real[:,1:1+x_target_labels.shape[1]], x_target_labels))
        loss_dis_user = F.average(loss_class(d_real[:,-user_label.shape[1]:], user_label))

        # loss weight 
        w_adv = isDiffTargetAndInput * float(batchsize) / (self.xp.sum(isDiffTargetAndInput.astype(float)) + 1e-6) # 変換元と変換先が異なるものに重みあり
        w_eq =  (1.0 - isDiffTargetAndInput) * float(batchsize) / (self.xp.sum(1.0 - isDiffTargetAndInput.astype(float)) + 1e-6) # 同じものに重みあり

        # adv loss 
        if self.loss_type == 'hinge':
            loss_dis_adv_fake = F.average(w_adv * loss_hinge_dis_fake(d_x_y))
            loss_dis_adv_real = F.average(w_adv * loss_hinge_dis_real(d_real))
            loss_dis_adv = loss_dis_adv_fake + loss_dis_adv_real
            loss_dis = self._lam_d_ad * loss_dis_adv + self._lam_d_ges * loss_dis_ges + self._lam_d_user * loss_dis_user

        elif self.loss_type == 'ls':        
            loss_dis_adv_fake = F.average(w_adv * loss_ls_dis_fake(d_x_y))
            loss_dis_adv_real = F.average(w_adv * loss_ls_dis_real(d_real))
            loss_dis_adv = loss_dis_adv_fake + loss_dis_adv_real
            loss_dis = self._lam_d_ad * loss_dis_adv + self._lam_d_ges * loss_dis_ges + self._lam_d_user * loss_dis_user

        elif self.loss_type == 'wgan-gp':
            loss_dis_adv = F.average(d_x_y[:,0] - d_real[:,0])
            # calcurate GP
            epsilon = self.xp.random.rand(batchsize,1,1,1).astype("f")
            y_hat = Variable(epsilon * x_y.data + (1-epsilon) * y_real.data)
            d_y_hat = self.dis(y_hat)
            g_d, = chainer.grad([w_adv * d_y_hat[:,:1]], [y_hat], enable_double_backprop=True)
            g_d_norm = F.sqrt(F.batch_l2_norm_squared(g_d) + 1e-6)
            loss_dis_gp = F.mean_squared_error(g_d_norm, self.xp.ones_like(g_d_norm.data))
            loss_dis_drift = F.average(d_real[:,0]*d_real[:,0])

            loss_dis = self._lam_d_ad * loss_dis_adv + self._lam_d_ges * loss_dis_ges + self._lam_d_user * loss_dis_user + self._lam_d_gp * loss_dis_gp + self._lam_d_drift * loss_dis_drift
            chainer.report({'loss_gp': self._lam_d_gp*loss_dis_gp}, self.dis)
            chainer.report({'loss_drift': self._lam_d_drift*loss_dis_drift}, self.dis)

        else:
            print(f'invalid loss type!!! ({self.loss_type})')
            assert False

        chainer.report({'loss_adv': self._lam_d_ad*loss_dis_adv}, self.dis)
        # chainer.report({'loss_cls': self._lam_d_cls*loss_dis_cls}, self.dis)
        chainer.report({'loss_ges': self._lam_d_ges*loss_dis_ges}, self.dis)
        chainer.report({'loss_user': self._lam_d_user*loss_dis_user}, self.dis)
        self.dis.cleargrads()
        loss_dis.backward()
        opt_dis.update()




        ## update Generator
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

        # eq loss
        if self.criterion == 'l2':
            loss_gen_equal = F.average(w_eq * F.average(F.squared_error(x_y, x), axis=(1,2,3)))
        elif self.criterion == 'l1':
            loss_gen_equal = F.average(w_eq * F.average(F.absolute_error(x_y, x), axis=(1,2,3)))

        # smooth loss
        # loss_gen_sm = 0
        # for i in range(10):
        #     loss_gen_sm += F.mean_absolute_error(x_y[:,:,i+1:,:], x_y[:,:,:-i-1,:]) / (i+1)
        loss_gen_sm = F.mean_absolute_error(x_y[:,:,1:,:], x_y[:,:,:-1,:])

        # class loss
        # loss_gen_cls = F.average(loss_class(d_x_y2[:,1:], x_target_labels)) # 変換後データが変換後ラベルと分類されるように学習
        # loss_gen_cls = F.average(loss_class(d_x_y2[:,1:], F.concat([x_target_labels, user_label], axis=1)))
        loss_gen_ges = F.average(loss_class(d_x_y2[:,1:1+x_target_labels.shape[1]], x_target_labels))
        loss_gen_user = F.average(loss_class(d_x_y2[:,-user_label.shape[1]:], user_label))
 
        # cyclic loss
        if self.criterion == 'l2':
            loss_cycle = F.mean_squared_error(x_y_x, x) if self._lam_g_rec > 0 else 0
        elif self.criterion == 'l1':
            loss_cycle = F.mean_absolute_error(x_y_x, x) if self._lam_g_rec > 0 else 0

        if self.cfg.train.class_equal:
            loss_gen = self._lam_g_ad * loss_gen_adv + self._lam_g_eq * loss_gen_equal + self._lam_g_sm * loss_gen_sm + self._lam_g_ges * loss_gen_ges + self._lam_g_user * loss_gen_user + self._lam_g_rec * loss_cycle
            # loss_gen = self._lam_g_ad * loss_gen_adv + self._lam_g_eq * loss_gen_equal + self._lam_g_cls * (loss_gen_cls+loss_gen_usercls) + self._lam_g_rec * loss_cycle
        else:
            loss_gen = self._lam_g_ad * loss_gen_adv + self._lam_g_sm * loss_gen_sm + self._lam_g_ges * loss_gen_ges + self._lam_g_user * loss_gen_user + self._lam_g_rec * loss_cycle


        if loss_dis_adv.data < 0.5 * loss_gen_adv.data:
            n_gen = 5
        else:
            n_gen = 1
        
       
        for i_ in range(n_gen):
            self.gen.cleargrads()
            loss_gen.backward()
            opt_gen.update()

        chainer.report({'loss_rec': loss_cycle * self._lam_g_rec},  self.gen)
        if self.cfg.train.class_equal:
            chainer.report({'loss_eq': loss_gen_equal * self._lam_g_eq},  self.gen)
        chainer.report({'loss_sm': loss_gen_sm * self._lam_g_sm},  self.gen)
        chainer.report({'loss_adv': loss_gen_adv * self._lam_g_ad}, self.gen)
        # chainer.report({'loss_cls': loss_gen_cls * self._lam_g_cls}, self.gen)
        chainer.report({'loss_ges': loss_gen_ges * self._lam_g_ges}, self.gen)
        chainer.report({'loss_user': loss_gen_user * self._lam_g_user}, self.gen)
        chainer.report({'lr': opt_gen.alpha})


        if self._iter % self.preview_interval == 0:
            save_path = os.path.join(self.save_dir, 'preview')
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            x = chainer.backends.cuda.to_cpu(x.data)
            x_y = chainer.backends.cuda.to_cpu(x_y.data)
            x_y_x = chainer.backends.cuda.to_cpu(x_y_x.data)
            y_real = chainer.backends.cuda.to_cpu(y_real.data)
            user_list = np.concatenate([x, x_y, x_y_x, y_real], axis=1)

            np.save(os.path.join(save_path, f'iter_{self._iter:04d}'), user_list)

            with open(os.path.join(save_path, 'preview.txt'), 'a') as f:
                source = np.where(chainer.backends.cuda.to_cpu(labels)==1)[-1]
                target = np.where(chainer.backends.cuda.to_cpu(target_labels)==1)[-1]
                user = np.where(chainer.backends.cuda.to_cpu(user_id)==1)[-1]
                f.write(f'iter {self._iter:04d}\n')
                for b in range(batchsize):
                    f.write(f'{b}: {source[b]} -> {target[b]} ({user[b]})\n')
                f.write('\n')
            

        # if self._iter % self.preview_interval == 0:
        #     for t in range(3):
        #        if self._lam_g_rec > 0:
        #            motion_list = [x[t,:,:,:], x_y[t,:,:,:], x_y_x[t,:,:,:], y_real[t,:,:,:]]
        #        else: 
        #            motion_list = [x[t,:,:,:], x_y[t,:,:,:], y_real[t,:,:,:]]
        #        self.save_images(motion_list, os.path.join(self.save_dir, 'preview', f'iter_{self._iter:09d}_{t+1}'))


    # def save_images(self, x_list, path):
    #     if not os.path.exists(os.path.split(path)[0]):
    #         os.makedirs(os.path.split(path)[0])

    #     positions_list = []
    #     for x in x_list:
    #         x = chainer.backends.cuda.to_cpu(x.data)
    #         x = np.reshape(np.asarray(x, dtype='float32'), (x.shape[1], -1))

    #         # add global position
    #         x = np.concatenate((np.zeros((x.shape[0], 3)), x), axis=1)

    #         for j in range(3, x.shape[1]):
    #             if self.datamax[j] - self.datamin[j] > 0:
    #                 x[:,j] = x[:,j] * (self.datamax[j] - self.datamin[j]) + self.datamin[j]

    #         positions = np.reshape(x, (x.shape[0], x.shape[1]//3, 3))

    #         n = Normalize()
    #         # 適当なbvhからoffsetをとる
    #         offsets = n.parse_initial_pos('data/bvh/CMU_jp/Locomotion_jp/walking_jp/02_01.bvh')

    #         #normalizeを戻す
    #         root_norm_positions = n.zero_add(positions, offsets)
    #         rec_norm_positions = n.convert_to_relative(root_norm_positions)
    #         denorm_relative_positions = n.denormalization_by_offset(rec_norm_positions, offsets)
    #         global_positions = n.convert_to_global(denorm_relative_positions, denorm_relative_positions[:,0,:])
    #         positions_list.append(global_positions)
    #     show_pose_multi(positions_list, path + '.gif', delay=self.cfg.train.frame_step, nogrid=False, lw=6)

