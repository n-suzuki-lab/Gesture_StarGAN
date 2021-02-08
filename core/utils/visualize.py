#!/usr/bin/env python
import os
import sys
import numpy as np
import chainer
import chainer.backends.cuda
from chainer import Variable
import chainer.functions as F

from core.utils.XYZnormalize import Normalize
from core.utils.make_gif_fromposition import show_pose_multi

def out_generated_image_jp(gen, dataset, depth, frame_length, datamin, datamax, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        xp = gen.xp
        z1 = Variable(xp.zeros(gen.make_hidden(1).shape).astype('float32'))
        z2 = Variable(xp.asarray(gen.make_hidden(1)))
        z3 = Variable(xp.asarray(gen.make_hidden(1)))

        batch = dataset[np.random.randint(0,len(dataset)-1)]
        x_real_data =  xp.array([batch])
        x_real = Variable(x_real_data.astype('float32'))

        keypose = F.concat((F.reshape(x_real[:,:,0,:],(x_real.shape[0], x_real.shape[1], 1, x_real.shape[3])),
                            F.reshape(x_real[:,:,-1,:], (x_real.shape[0], x_real.shape[1], 1, x_real.shape[3]))), axis=2)

        with chainer.using_config('train', False):
            x1 = gen(keypose, z1, 1.0)
        with chainer.using_config('train', False):
            x2 = gen(keypose, z2, 1.0)
        with chainer.using_config('train', False):
            x3 = gen(keypose, z3, 1.0)

        preview_dir = '{0}/preview/depth{1}'.format(dst, depth)
        preview_path_gif = preview_dir +\
            '/image{:0>8}'.format(trainer.updater.iteration)
        preview_path1 = preview_dir +\
            '/image{:0>8}_1'.format(trainer.updater.iteration)
        preview_path2 = preview_dir +\
            '/image{:0>8}_2'.format(trainer.updater.iteration)
        preview_path3 = preview_dir +\
            '/image{:0>8}_3'.format(trainer.updater.iteration)
        preview_path_real = preview_dir +\
            '/image{:0>8}_real'.format(trainer.updater.iteration)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)

        global_x1 = save_gif(x1, preview_path1, depth, datamin, datamax)
        global_x2 = save_gif(x2, preview_path2, depth, datamin, datamax)
        global_x3 = save_gif(x3, preview_path3, depth, datamin, datamax)
        global_x_real = save_gif(x_real, preview_path_real, depth, datamin, datamax)

        positions_list = [global_x1, global_x2, global_x3, global_x_real]
        show_pose_multi(positions_list, preview_path_gif + '.gif', 64 // (2**(depth+1)))
    return make_image

def out_generated_image_nokey_jp(gen, dataset, depth, frame_length, datamin, datamax, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        xp = gen.xp
        z1 = Variable(xp.asarray(gen.make_hidden(1)))
        z2 = Variable(xp.asarray(gen.make_hidden(1)))
        z3 = Variable(xp.asarray(gen.make_hidden(1)))

        batch = dataset[np.random.randint(0,len(dataset)-1)]
        x_real_data =  xp.array([batch])
        x_real = Variable(x_real_data.astype('float32'))

        with chainer.using_config('train', False):
            x1 = gen(z1, 1.0)
        with chainer.using_config('train', False):
            x2 = gen(z2, 1.0)
        with chainer.using_config('train', False):
            x3 = gen(z3, 1.0)

        preview_dir = '{0}/preview/depth{1}'.format(dst, depth)
        preview_path_gif = preview_dir +\
            '/image{:0>8}'.format(trainer.updater.iteration)
        preview_path1 = preview_dir +\
            '/image{:0>8}_1'.format(trainer.updater.iteration)
        preview_path2 = preview_dir +\
            '/image{:0>8}_2'.format(trainer.updater.iteration)
        preview_path3 = preview_dir +\
            '/image{:0>8}_3'.format(trainer.updater.iteration)
        preview_path_real = preview_dir +\
            '/image{:0>8}_real'.format(trainer.updater.iteration)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)

        global_x1 = save_gif(x1, preview_path1, depth, datamin, datamax)
        global_x2 = save_gif(x2, preview_path2, depth, datamin, datamax)
        global_x3 = save_gif(x3, preview_path3, depth, datamin, datamax)
        global_x_real = save_gif(x_real, preview_path_real, depth, datamin, datamax)

        positions_list = [global_x1, global_x2, global_x3, global_x_real]
        show_pose_multi(positions_list, preview_path_gif + '.gif', 64 // (2**(depth)))
    return make_image



def save_gif(positions, path, depth, datamin, datamax):
    positions = chainer.backends.cuda.to_cpu(positions.data)
    positions = np.reshape(np.asarray(positions, dtype='float32'), (2**(depth+1), -1))
    for j in range(positions.shape[1]):
        if datamax[j] - datamin[j] > 0:
            positions[:,j] = positions[:,j] * (datamax[j] - datamin[j]) + datamin[j]

    positions = np.reshape(positions, (positions.shape[0], positions.shape[1]//3, 3))

    np.save(path+'.npy', positions)


    n = Normalize()
    # 適当なbvhからoffsetをとる
    offsets = n.parse_initial_pos('data/bvh/CMU_jp/Locomotion_jp/walking_jp/02_01.bvh')

    #normalizeを戻す
    root_norm_positions = n.zero_add(positions, offsets)
    rec_norm_positions = n.convert_to_relative(root_norm_positions)
    denorm_relative_positions = n.denormalization_by_offset(rec_norm_positions, offsets)
    global_positions = n.convert_to_global(denorm_relative_positions, denorm_relative_positions[:,0,:])

    return global_positions
