#!/usr/bin/env python
import argparse
import os, sys
import shutil
from pathlib import Path

import numpy as np
import pickle
import chainer
from chainer import Variable, cuda
from chainer import training, serializers
from chainer.training import extensions

from core import models
from core.utils.config import Config
#from core.utils.tensorboardreport import TensorBoardReport
from core.updater.EqualledArgumentedCycleGANupdater import EqualledArgumentedCycleGANUpdater
# from core.datasets.dataset import data_load


def parse_args():
    parser = argparse.ArgumentParser(description='PRGAN')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    return args

## Data collector and crawler function
def data_load(path, gesture_num=4, skip=None, datamin=0, datamax=1023):
    gestures = []
    for i in range(gesture_num):
        if i == skip:
            continue
        gesture = np.load(f'{path}/gesture{i+1}.npy')[:,20:,:]
        gestures.append((gesture-datamin)/(datamax-datamin))

    return gestures


def test():
    args = parse_args()
    cfg = Config.from_file(args.config)

    data_path = Path(__file__).parent / '..' / cfg.test.dataset
    out = cfg.test.out
    n_class = len(cfg.train.dataset_dirs)
    rep = cfg.train.iterations // cfg.train.save_interval
    model_path = cfg.test.gen

    print('GPU: {}'.format(args.gpu))

    gesture_list = data_load(data_path)
    test_data = np.concatenate(gesture_list)
    source_label = cfg.source
    target_label = cfg.target
    x_data = np.expand_dims(test_data, axis=1)
    labels = np.zeros((x_data.shape[0], n_class*2, 1, 1))
    labels[:,source_label] += 1
    labels[:,target_label+n_class] += 1

    x_data = Variable(cuda.to_gpu(x_data.astype(np.float32)))
    labels = Variable(cuda.to_gpu(labels.astype(np.float32)))

    # for i in range(rep):
    for i in range(1):
        ## Set up a neural network to train
        ## "n_class" is number of styles (i.e. Old, Normal, ...)
        gen = getattr(models, cfg.train.generator.model)(cfg.train.generator, n_class=n_class) 
            
        ## Load resume checkpoint (only weight, not load optimizer state)
        # gen_weight = cfg.test.gen
        # iter = (i+1) * cfg.train.save_interval
        iter = cfg.train.iterations
        gen_weight = model_path + f'/gen_iter_{iter}.npz'
        print('')
        print(f'loading generator weight from {gen_weight}')
        serializers.load_npz(gen_weight, gen)

        ## Set GPU
        if args.gpu >= 0:
            # Make a specified GPU current
            chainer.backends.cuda.get_device_from_id(args.gpu).use()
            gen.to_gpu()

        ## Save scripts and command to result directory. (To look back later.)
        # if not os.path.exists(out):
        #     os.makedirs(out)

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            gen_data = gen(x_data, labels)
        gen_data.to_cpu()
        print(gen_data.shape)

        save_path = f'./{out}{iter}'
        print('saving generated data to ' + save_path + '.npy')
        np.save(save_path, gen_data.data)


if __name__ == '__main__':
    test()
