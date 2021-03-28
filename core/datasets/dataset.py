import os, sys, glob
import random

import pickle
import numpy as np
import chainer
from chainer import datasets


## Data loader
# return data (dict{user_id:data}) and data length (dict{user_id:length})
# please edit to fit your data.
def data_load(path, n_gesture=4, datamin=0, datamax=1):
    gestures = []
    length = []
    for i in range(n_gesture):
        gesture = np.load(f'{path}/gesture{i+1}.npy')
        gestures.append((gesture-datamin)/(datamax-datamin)) # normalization
        length.append(gesture.shape[0])

    return gestures, length

## Dataset Class
class GestureDataset(chainer.dataset.DatasetMixin):
    def __init__(self, data_dirs, style='gesture', equal=False, datamin=0, datamax=1): 
        # * Argument
        #   - dataset_dirs : List of root directory which contains each style gesture datasets.
        #   - datamin, datamax: min and max value of data for normalization. When not normalizing, set 0 and 1.
        #   - equal : Whether to use 'equall loss' in train. If it's true, data loader creates batch including some pairs of same style class.
        n_user = len(data_dirs)
        n_gesture = 4

        self.style = style
        self.n_style = n_gesture if style == 'gesture' else n_user
        self.n_content = n_user if style == 'gesture' else n_gesture

        self.datalist_list = []
        self.datalen_list = []
        self.equal = equal
        for data_dir in data_dirs:
            gestures, data_len = data_load(data_dir, datamin=datamin, datamax=datamax)
            self.datalist_list.append(gestures)
            self.datalen_list.append(data_len)

    def __len__(self):
        return sum(map((lambda l: sum(l)), self.datalen_list))

    def len_each(self):
        return self.datalen_list

    def get_example(self, i, whichClass=None):
        # If whichClass is set as a particular class, load that class data as sample x.
        # Otherwise, load random class data. 
        # If equal is True, class of sample y is randomly chosen so that it is the same class by 50% and a different class by 50%.
        x_class = whichClass if whichClass else np.random.randint(self.n_style)
        if self.equal:
             if np.random.randint(2) == 0:
                 # 50% -> same class
                  y_class = x_class 
             else:
                 # 50% -> diffrent class
                 y_class = np.random.randint(self.n_style-1)
                 if y_class >= x_class: y_class += 1
        else:
            y_class = np.random.randint(self.n_style-1)
            if y_class >= x_class: y_class += 1

        content_class = np.random.randint(self.n_content)
        content_label = np.zeros((1, self.n_content))
        content_label[:,content_class] += 1

        # choose a sample randomly from x_class and y_class
        if self.style == 'gesture':
            x_index = np.random.randint(self.datalen_list[content_class][x_class])
            x_data = self.datalist_list[content_class][x_class][x_index]
        
            y_index = np.random.randint(self.datalen_list[content_class][y_class])
            y_data = self.datalist_list[content_class][y_class][y_index]
            
        else:
            x_index = np.random.randint(self.datalen_list[x_class][content_class])
            x_data = self.datalist_list[x_class][content_class][x_index]
            
            y_index = np.random.randint(self.datalen_list[y_class][content_class])
            y_data = self.datalist_list[y_class][content_class][y_index]

        x_label = np.zeros((1,self.n_style))
        x_label[:,x_class] += 1
        y_label = np.zeros((1,self.n_style))
        y_label[:,y_class] += 1

        return x_data, x_label, y_data, y_label, content_label
