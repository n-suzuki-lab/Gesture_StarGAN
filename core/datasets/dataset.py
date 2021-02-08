import os, sys, glob
import random

import pickle
import numpy as np
import chainer
from chainer import datasets


## Data collector and crawler function
def data_load(path, gesture_num=4, datamin=0, datamax=1023):
    gestures = {}
    length = {}
    # ges_label = set()
    for i in range(gesture_num):
        gesture = np.load(f'{path}/gesture{i+1}.npy')[:,20:,:]
        gestures[i] = (gesture-datamin)/(datamax-datamin)
        length[i] = gesture.shape[0]

    return gestures, length

## Dataset Class
class MultiDataset_jp_GAN(chainer.dataset.DatasetMixin):
    def __init__(self, data_dirs, datamin=0, datamax=1023, equal=False):
        self.n_user = len(data_dirs)
        self.n_class = 4
        self.datalist_list = []
        self.datalen_list = []
        # self.geslabel_list = []
        self.equal = equal
        for i in range(self.n_user):
            gestures, data_len = data_load(data_dirs[i])
            self.datalist_list.append(gestures)
            self.datalen_list.append(data_len)
            # self.geslabel_list.append(ges_label)

    def __len__(self):
        return sum(map((lambda dic: sum(list(dic.values()))), self.datalen_list))

    def len_each(self):
        return self.datalen_list

    def get_example(self, i, whichClass=None):
        # Index list of frames to extract according to frame_step.
        # ex = [k * self.frame_step for k in range(self.frame_nums)]

        # If whichClass is set as a particular class, load that class data as sample x.
        # Otherwise, load random class data. 
        # Then, class of sample y is randomly chosen so that it is the same class by 50% and a different class by 50%.
        x_class = whichClass if whichClass else np.random.randint(self.n_class)
        if self.equal:
             if np.random.randint(2) == 0:
                 # 50% -> same class
                  y_class = x_class 
             else:
                 # 50% -> diffrent class
                 y_class = np.random.randint(self.n_class-1)
                 if y_class >= x_class: y_class += 1
        else:
            y_class = np.random.randint(self.n_class-1)
            if y_class >= x_class: y_class += 1

        user_id = np.random.randint(self.n_user)

        x_index = np.random.randint(self.datalen_list[user_id][x_class])
        x_data = self.datalist_list[user_id][x_class][x_index]
        x_label = np.zeros((1,self.n_class))
        x_label[:,x_class] += 1
        
        y_index = np.random.randint(self.datalen_list[user_id][y_class])
        y_data = self.datalist_list[user_id][y_class][y_index]
        y_label = np.zeros((1,self.n_class))
        y_label[:,y_class] += 1

        user_label = np.zeros((1, self.n_user))
        user_label[:,user_id] += 1

        return x_data, x_label, y_data, y_label, user_label
