import skimage.io as io
import skimage.transform as transform
import itertools
from matplotlib import pyplot as plt
import numpy as np
import copy
import os
import gc

#load images 
def load(path, folder = 'dat_bin', s_name = 'def', res = [256, 256], sv_bin = False, chunk_size = 1000):
    namelist = os.listdir(path)
    imgs = []
    count = 0
    for name in namelist:
        try:
            img = io.imread(path + '/' + name)
            if(img.shape[:-1] != res):
                img = transform.resize(img, output_shape = res)
            imgs.append(img)
        except:
            pass
        gc.collect()
        if sv_bin:
            count += 1
            print(count)
            if count > 0 and count % chunk_size == 0:
                file = os.path.abspath('%s/%s_%i.npy' % (folder, s_name, count // chunk_size))
                np.save(file, np.array(imgs))
                imgs = []
    if not sv_bin:        
        return np.array(imgs)
    else:
        file = os.path.abspath('%s/%s_%i.npy' % (folder, s_name, count // chunk_size))
        np.save(file, np.array(imgs))
        imgs = []

def load_chunk(path):
    return np.load(path)

def display(imgs, res = [64, 64], grid_size = 5, fig_size = [5,5], show = False, save = False, folder = 'test_fl', file_name = 'fig.png'):
    fig, ax = plt.subplots(grid_size, grid_size, figsize = fig_size)

    for i, j in itertools.product(range(grid_size), range(grid_size)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(grid_size ** 2):
        i = k // grid_size
        j = k % grid_size
        ax[i, j].imshow(transform.resize(imgs[k], output_shape = res))
    
    if show:
        plt.show()
    if save:
        try:
            plt.savefig(os.path.abspath('%s/%s' % (folder, file_name)))
        except:
            os.mkdir(os.path.abspath(folder))
            plt.savefig(os.path.abspath('%s/%s' % (folder, file_name)))

class ItemsPool():
    
    def __init__(self, max_items = 50):
        self.max_items = max_items 
        self.num = 0 
        self.items = []

    def __call__(self, _items):
        if self.max_items == 0:
            return _items
        return_items = []
        for _item in _items:
            if self.num < self.max_items:
                self.items.append(_item)
                self.num += 1
                return_items.append(_item)
            else:
                if np.random.rand() > 0.5:
                    ind = np.random.randint(0, self.max_items)
                    temp = copy.copy(self.items[ind])
                    self.items[ind] = _item
                    return_items.append(temp)
                else:
                    return_items.append(_item)
        return return_items

if __name__ == '__main__':
    try:
        os.listdir('dat_bin')
    except:
        os.mkdir('dat_bin')
    
    load(path = os.path.abspath('Datasets/face'), s_name = 'dat',res = [64, 64, 4], sv_bin = True)