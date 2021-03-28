#!/usr/bin/env python
import argparse
import os, sys
import shutil
from pathlib import Path

import numpy as np
import pickle
import chainer
from chainer import training, serializers
from chainer.training import extensions

from core import models
from core.utils.config import Config
from core.updater.StarGANupdater import StarGANUpdater
from core.datasets.dataset import GestureDataset


def parse_args():
    parser = argparse.ArgumentParser(description='StarGAN')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    return args


def train():
    args = parse_args()
    cfg = Config.from_file(args.config)

    iteration = cfg.train.iterations
    batchsize = cfg.train.batchsize
    out = cfg.train.out

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(batchsize))
    print('# iteration: {}'.format(iteration))
    print('')

    ## Set up networks to train
    # number of gesture classes and users
    n_gesture = cfg.train.n_gesture
    n_user = len(cfg.train.dataset_dirs)
    if cfg.style == 'gesture':
        gen = getattr(models, cfg.train.generator.model)(cfg.train.generator, n_style=n_gesture) 
    elif cfg.style == 'user':
        gen = getattr(models, cfg.train.generator.model)(cfg.train.generator, n_style=n_user)
    else:
        print(f'Invalid style: {cfg.style}')
        exit()
    dis = getattr(models, cfg.train.discriminator.model)(cfg.train.discriminator, n_gesture=n_gesture, n_user=n_user)
 
    ## Load resume checkpoint to restart. Load optimizer state later.
    if args.resume or cfg.resume:
        gen_resume =  args.resume if args.resume else cfg.resume
        print(f'loading generator resume from {os.path.join(out, gen_resume)}')
        serializers.load_npz(os.path.join(out, gen_resume), gen)
        dis_resume = gen_resume.replace('gen', 'dis')
        print(f'loading discriminator resume from {os.path.join(out, dis_resume)}')
        serializers.load_npz(os.path.join(out, dis_resume), dis)
        
    ## Load resume checkpoint (only weight, not load optimizer state)
    if cfg.weight:
        gen_weight = cfg.weight
        print(f'loading generator weight from {gen_weight}')
        serializers.load_npz(gen_weight, gen)
        dis_weight = gen_weight.replace('gen', 'dis')
        print(f'loading discriminator weight from {dis_weight}')
        serializers.load_npz(dis_weight, dis)
            
 
    ## Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        return optimizer
    opt_gen = make_optimizer(gen, alpha=cfg.train.parameters.g_lr, beta1=0.5)
    opt_dis = make_optimizer(dis, alpha=cfg.train.parameters.d_lr, beta1=0.5)
    if args.resume or cfg.resume:
        opt_gen_resume = gen_resume.replace('gen', 'opt_gen')
        print(f'loading generator optimizer from {os.path.join(out, opt_gen_resume)}')
        serializers.load_npz(opt_gen_resume, opt_gen)
        opt_dis_resume = gen_resume.replace('gen', 'opt_dis')
        print(f'loading discriminator optimizer from {os.path.join(out, opt_dis_resume)}')
        serializers.load_npz(opt_dis_resume, opt_dis)

    ## Set GPU
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()

    ## Set up dataset
    train = GestureDataset(cfg.train.dataset_dirs, style=cfg.style, equal=cfg.train.class_equal)
    for i in range(len(cfg.train.dataset_dirs)):
        print(f'{cfg.train.dataset_dirs[i]} contains {train.len_each()[i]} samples')

    train_iter = chainer.iterators.SerialIterator(train, batchsize)

    ## Set up a Trainer
    updater = StarGANUpdater(
        models = (gen, dis),
        iterator = train_iter,
        optimizer = {'gen': opt_gen, 'dis': opt_dis},
        cfg = cfg,
        out=out)
    trainer = training.Trainer(updater, (iteration, 'iteration'), out=out)

    ## Set invervals
    ##     - display_interval : Interval iterations of print logs on display.
    ##     - save_interval : Interval iterations of save models and preview images.
    display_interval = (cfg.train.display_interval, 'iteration')
    save_interval = (cfg.train.save_interval, 'iteration')
    trainer.extend(extensions.snapshot_object(gen, 'gen_iter_{.updater.iteration}.npz'), trigger=save_interval )
    trainer.extend(extensions.snapshot_object(dis, 'dis_iter_{.updater.iteration}.npz'), trigger=save_interval )
    trainer.extend(extensions.snapshot_object(opt_gen, 'opt_gen_iter_{.updater.iteration}.npz'), trigger=save_interval )
    trainer.extend(extensions.snapshot_object(opt_dis, 'opt_dis_iter_{.updater.iteration}.npz'), trigger=save_interval )
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
            'epoch', 'iteration', 'lr', 'gen/loss_adv', 'gen/loss_eq', 'gen/loss_style', 'gen/loss_cont', 'gen/loss_rec', 'gen/loss_sm', 'dis/loss_adv', 'dis/loss_style', 'dis/loss_cont' 
        ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=display_interval[0]))

    ## Save scripts and command to result directory. (To look back later.)
    if not os.path.exists(out):
        os.makedirs(out)
    shutil.copy(args.config, f'./{out}')
    shutil.copy('./core/models/StarGAN.py', f'./{out}')
    shutil.copy('./core/updater/StarGANupdater.py', f'./{out}')

    commands = sys.argv
    with open(f'./{out}/command.txt', 'w') as f:
        f.write(f'python {commands[0]} ')
        for command in commands[1:]:
            f.write(command + ' ')

    ## Run the training
    trainer.run()

    ## Once training finishid, save all models.
    modelname = f'./{out}/gen.npz'
    print('saving generator model to ' + modelname)
    serializers.save_npz(modelname, gen)

    modelname = f'./{out}/dis.npz'
    print('saving discriminator model to ' + modelname)
    serializers.save_npz(modelname, dis)

    optname = f'./{out}/opt_gen.npz'
    print( 'saving generator optimizer to ' + optname)
    serializers.save_npz(optname, opt_gen)

    optname = f'./{out}/opt_dis.npz'
    print( 'saving discriminator optimizer to ' + optname)
    serializers.save_npz(optname, opt_dis)


if __name__ == '__main__':
    train()
