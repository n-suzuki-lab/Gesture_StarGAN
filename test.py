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


def parse_args():
    parser = argparse.ArgumentParser(description='PRGAN')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    return args

## Data loader
# please edit to fit your data.
def data_load(path, ges_class=0, datamin=0, datamax=1023):
    gesture = np.load(f'{path}/gesture{ges_class+1}.npy')[:,20:,:]
    return (gesture-datamin)/(datamax-datamin)

def test():
    args = parse_args()
    cfg = Config.from_file(args.config)

    data_path = cfg.test.source_data
    out = cfg.test.out
    model_path = cfg.test.gen_path

    print('GPU: {}'.format(args.gpu))

    # number of gesture classes and users
    n_gesture = cfg.train.n_gesture
    n_user = len(cfg.train.dataset_dirs)
    n_style = n_gesture if cfg.style == 'gesture' else n_user

    ## Import generator model
    gen = getattr(models, cfg.train.generator.model)(cfg.train.generator, n_style=n_style) 
    serializers.load_npz(model_path, gen)
    print('')
    print(f'loading generator weight from {model_path}')

    ## Set GPU
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()

    test_data = data_load(data_path, ges_class=cfg.test.ges_class)
    test_data = np.expand_dims(test_data, axis=1)

    style_label = np.zeros((test_data.shape[0], n_style*2, 1, 1))
    source = cfg.test.ges_class if cfg.style == 'gesture' else cfg.source_user
    target = cfg.test.target_style
    style_label[:,source] += 1
    style_label[:,target+n_style] += 1

    test_data = Variable(cuda.to_gpu(test_data.astype(np.float32)))
    style_label = Variable(cuda.to_gpu(style_label.astype(np.float32)))

    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        gen_data = gen(test_data, style_label)
    gen_data.to_cpu()

    if cfg.style == 'gesture':
        save_path = f'./{out}/user{cfg.source_user}/ges{target}_from_ges{source}'
    else:
        save_path = f'./{out}/user{target}/ges{cfg.test.ges_class}_from_user{source}'
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    print('saving generated data to ' + save_path + '.npy')
    np.save(save_path, gen_data.data)


if __name__ == '__main__':
    test()
