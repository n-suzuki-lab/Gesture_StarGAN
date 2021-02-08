import os
import sys
import glob

import numpy as np
from sklearn.gaussian_process import kernels as sk_kern
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from XYZnormalize import Normalize
from make_gif_fromposition import show_pose_multi

def loadData(npy_path, minmax_path, save_dir):
    assert os.path.exists(npy_path)
    assert os.path.exists(minmax_path)

    with np.load(minmax_path) as data:
        datamin = data['min']
        datamax = data['max']

    motion = np.load(npy_path)[::8,:]
    save_image(motion, save_dir+'/motion')
    motion = np.reshape(motion, (motion.shape[0], motion.shape[1] * 3))
    motion = motion[:,3:]
    #for j in range(motion.shape[1]):
    #    if datamax[j] - datamin[j] > 0:
    #        motion[j,:] = (motion[j,:] - datamin[j]) / (datamax[j] - datamin[j])

    return motion

def GPreg(motion, save_dir):
    # kernel
    kernel = sk_kern.RBF(1.0, (1e-3, 1e3)) + sk_kern.ConstantKernel(1.0, (1e-3, 1e3)) + sk_kern.WhiteKernel()
    clf = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=20,
            normalize_y=False)

    for joint in range(27):
        for dim in range(0,3):
            x_train = np.array(list(range(motion.shape[0])))
            y_train = np.array([motion[t,joint*3+dim] for t in range(motion.shape[0])])

            clf.fit(x_train.reshape(-1,1), y_train)
            clf.kernel_

            x_test = np.linspace(0, motion.shape[0], 3000).reshape(-1,1)

            pred_mean = clf.predict(x_test)

            plt.plot(x_test[:,0], pred_mean, label=f'Joint:{joint}')

        plt.legend()
        plt.show()
        plt.savefig(save_dir+f'/joint_{joint}.png')
        plt.clf()

def save_image(positions, path):
    positions_list = []

    n = Normalize()
    # 適当なbvhからoffsetをとる
    offsets = n.parse_initial_pos('data/bvh/CMU_jp/Locomotion_jp/walking_jp/02_01.bvh')

    #normalizeを戻す
    root_norm_positions = n.zero_add(positions, offsets)
    rec_norm_positions = n.convert_to_relative(root_norm_positions)
    denorm_relative_positions = n.denormalization_by_offset(rec_norm_positions, offsets)
    global_positions = n.convert_to_global(denorm_relative_positions, denorm_relative_positions[:,0,:])
    positions_list.append(global_positions)
    show_pose_multi(positions_list, path + '.gif', delay=8, nogrid=False, lw=6)




if __name__ == '__main__':
    npy_paths = glob.glob('data/train_jp/CMU_jp/Styled_jp/Subject137/*_Walkn.npy')
    minmax_path = 'data/minmax/minmax_Subject137_lotated.npz'
    result_dir = 'regression'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    for npy_path in npy_paths:
        motion_name = os.path.splitext(os.path.split(npy_path)[1])[0]
        print(f'Processing {motion_name}...')
        save_dir = os.path.join(result_dir, motion_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        motion = loadData(npy_path, minmax_path, save_dir)
        GPreg(motion, save_dir)
