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
        gestures.append(np.expand_dims((gesture-datamin)/(datamax-datamin), axis=1))

    return gestures


def test():
    args = parse_args()
    cfg = Config.from_file(args.config)

    data_path = 'data/data'
    # out = 'results/EqualledArgumentedCycleGAN/trial{trial}/without_ges{skip_ges}/'
    n_class = 10
    n_ges = 4
    model_path = 'results/EqualledArgumentedCycleGAN/trial42'

    print('GPU: {}'.format(args.gpu))

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()

    gens = []
    for ges in range(n_ges):
        gen = getattr(models, cfg.train.generator.model)(cfg.train.generator, n_class=n_class)
        # print('')
        # print(f'loading generator weight from {model_path}')
        serializers.load_npz(f'{model_path}/without_ges{ges}/gen_iter_80000.npz', gen)
        gens.append(gen.to_gpu())
        print(f'loading generator weight from {model_path}/without_ges{ges}/gen_iter_80000.npz')

    source_users = [0,1,2,3,7,8]
    target_users = [4,5,6,9]

    for source in source_users:
        source_ges = data_load(Path(__file__).parent / '..' / f'{data_path}/user{source+1:02}/first')

        for target in target_users:
            for ges in range(n_ges):
                x_data = source_ges[ges]
                labels = np.zeros((x_data.shape[0], n_class*2, 1, 1))
                labels[:,source] += 1
                labels[:,target+n_class] += 1

                x_data = Variable(cuda.to_gpu(x_data.astype(np.float32)))
                labels = Variable(cuda.to_gpu(labels.astype(np.float32)))

                with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                    gen_data = gens[ges](x_data, labels)
                gen_data.to_cpu()

                save_path = f'{model_path}/generated/user{target}/from_user{source}/gesture{ges}'
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                print('saving generated data to ' + save_path + '.npy')
                np.save(save_path, gen_data.data)



if __name__ == '__main__':
    test()
