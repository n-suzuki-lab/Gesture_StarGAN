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
    n_class = 4
    model_path = cfg.test.gen

    print('GPU: {}'.format(args.gpu))

    gen = getattr(models, cfg.train.generator.model)(cfg.train.generator, n_class=n_class)
    print('')
    print(f'loading generator weight from {model_path}')
    serializers.load_npz(model_path, gen)

    ## Set GPU
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()

    gesture_list = data_load(data_path)
    test_data = np.concatenate(gesture_list)
    test_data = test_data.reshape((n_class, -1, 1, test_data.shape[1], test_data.shape[2]))

    for source in range(n_class):
        for target in range(n_class):
            if source == target:
                continue
            x_data = test_data[source]
            labels = np.zeros((x_data.shape[0], n_class*2, 1, 1))
            labels[:,source] += 1
            labels[:,target+n_class] += 1

            x_data = Variable(cuda.to_gpu(x_data.astype(np.float32)))
            labels = Variable(cuda.to_gpu(labels.astype(np.float32)))

            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                gen_data = gen(x_data, labels)
            gen_data.to_cpu()

            save_path = f'./{out}/from_ges{source}/gesture{target}'
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            print('saving generated data to ' + save_path + '.npy')
            np.save(save_path, gen_data.data)



if __name__ == '__main__':
    test()
