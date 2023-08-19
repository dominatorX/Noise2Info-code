from pathlib import Path
import numpy as np
from sklearn import model_selection
from PIL import Image
from random import randint
import os
from glob import glob
from natsort import natsorted


def train_gen():
    data_dir = 'Training/Data'

    img_list = natsorted(glob(os.path.join(data_dir, '*', '*.PNG')))
    noisy_files, clean_files = [], []
    for file_ in img_list:
        filename = os.path.split(file_)[-1]
        if 'GT' in filename:
            clean_files.append(file_)
        if 'NOISY' in filename:
            noisy_files.append(file_)
    train_numpy_gen(clean_files, noisy_files)


def test_gen():
    data_dir = 'Testing/Data'

    img_list = natsorted(glob(os.path.join(data_dir, 'groundtruth', '*.png')))
    print(len(img_list))
    for img in img_list:
        clean = Image.open(str(img))
        noisy = Image.open(str(img).replace('groundtruth', 'input'))
        try:
            if 0. in np.array(clean).std(axis=(0, 1)):
                print(img)
                continue
            output = np.stack((np.array(clean), np.array(noisy)), axis=-1)
            np.save('.'.join([str(img).replace(data_dir+'groundtruth', 'testout').split('.')[0], 'npy']),
                    output)
        except Exception as e:
            print(e)
            print(img)
            exit(0)


def train_numpy_gen(clean_files):
    output = []
    pixel_size = 2304
    for img in clean_files:
        clean = Image.open(str(img))
        shape = np.array(clean).shape
        noisy = Image.open(str(img).replace('GT', 'NOISY'))
        idx1, idx2 = randint(0, shape[0] - pixel_size), randint(0, shape[1] - pixel_size)
        try:
            if 0. in np.array(clean)[idx1:idx1+pixel_size, idx2:idx2+pixel_size].std(axis=(0, 1)):
                print(img)
                continue
            output.append(np.stack((np.array(clean)[idx1:idx1+pixel_size, idx2:idx2+pixel_size],
                                    np.array(noisy)[idx1:idx1+pixel_size, idx2:idx2+pixel_size]), axis=0))
        except Exception as e:
            print(img)

    np.save('train.npy', np.array(output))


if __name__ == '__main__':
    train_gen()
    test_gen()
