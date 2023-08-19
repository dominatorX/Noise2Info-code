import os, fnmatch, cv2

import numpy as np
from models import Noise2Same
from models import Noise2Info
# from models_keras import Noise2Info
from utils.evaluation_utils import normalize
import tensorflow as tf
from tifffile import imread
import random


seed = 666
os.environ['PYTHONHASHSEED'] = '1'
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

dataset = 'hanzi'   # choose the dataset from  hanzi, imagenet, bsd68, sidd
model_name = 'n2i'   # chooise the model from   n2i:Noise2Info,  n2s: Noise2Same


def PSNR(gt, img):
    mse = np.mean(np.square(gt - img))
    return 20 * np.log10(255) - 10 * np.log10(mse)


def norm(x):
    x = (x-x.min())/(x.max()-x.min())
    return x


if dataset == 'hanzi':
    noise_level = 3
    data_dir = 'Your/Data/Dir/'
    X = np.load(data_dir+'train.npy')[:, noise_level, ..., None]
    X_val = np.load(data_dir+'val.npy')[:, noise_level, ..., None]
    X = np.array([(x - x.mean())/x.std() for x in X]).astype('float32')
    X_val = np.array([(x - x.mean())/x.std() for x in X_val]).astype('float32')
    if model_name == 'n2s':
        sgm_loss = 1
        model = Noise2Same('trained_models/', 'Hanzi/N2S-Random', dim=2, in_channels=1, lmbd=sgm_loss*2)
    elif model_name == 'n2i':
        model = Noise2Info('trained_models/', 'Hanzi/N2I-Random', dim=2, in_channels=1)
    model.train(X, patch_size=[64, 64], validation=X_val, batch_size=64, steps=50000)

elif dataset == 'bsd68':
    data_dir = 'Your/Data/Dir/'
    X = np.load(data_dir+'train/DCNN400_train_gaussian25.npy')
    X_val = np.load(data_dir+'val/DCNN400_validation_gaussian25.npy')
    X = np.array([(x - x.mean())/x.std() for x in X])
    X_val = np.array([(x - x.mean())/x.std() for x in X_val]).astype('float32')
    if model_name == 'n2s':
        sgm_loss = 0.4567
        model = Noise2Same('trained_models/', 'BSD68/N2S-Random', dim=2, in_channels=1, lmbd=sgm_loss*2)
    elif model_name == 'n2i':
        model = Noise2Info('trained_models/', 'BSD68/N2I-Random', dim=2, in_channels=1)
    model.train(X[..., None], patch_size=[64, 64], validation=X_val[..., None], batch_size=64, steps=80000)

elif dataset == 'imagenet':
    data_dir = 'Your/Data/Dir/'
    X = np.load(data_dir+'train.npy')[:, 1]
    X_norm = np.array([(x - x.mean(axis=(0, 1), keepdims=True)) / x.std(axis=(0, 1), keepdims=True) for x in X])
    if model_name == 'n2s':
        sgm_loss = 0.9483
        model = Noise2Same('trained_models/', 'ImageNet/N2S-Donut', dim=2, in_channels=3, lmbd=sgm_loss*2, masking='donut')
    elif model_name == 'n2i':
        model = Noise2Info('trained_models/', 'ImageNet/N2I-Donut', dim=2, in_channels=3, masking='donut')
    model.train(X_norm.astype('float32'), patch_size=[64, 64], validation=None, batch_size=64, steps=50000)

elif dataset == 'sidd':
    data_dir = 'Your/Data/Dir/'
    X = np.load(data_dir+'train.npy')[:, 1].astype('float32')
    X_norm = X/255.0
    if model_name == 'n2s':
        sgm_loss = 0.074
        model = Noise2Same('trained_models/', 'SIDD/N2S-Random', dim=2, in_channels=3, lmbd=sgm_loss*2)
    elif model_name == 'n2i':
        model = Noise2Info('trained_models/', 'SIDD/N2I-Random', dim=2, in_channels=3)
    model.train(X_norm.astype('float32'), patch_size=[64, 64], validation=None, batch_size=64, steps=50000)
