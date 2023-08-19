import os, fnmatch, cv2
import numpy as np
from utils.evaluation_utils import get_scores, normalize
from tifffile import imread
from models import Noise2Same
from models import Noise2Info
# from models_keras import Noise2Info
os.environ['CUDA_VISIBLE_DEVICES']='2'


dataset = 'imagenet'
model_name = 'n2i'


def PSNR(gt, img):
    mse = np.mean(np.square(gt - img))
    return 20 * np.log10(255) - 10 * np.log10(mse)

def norm(x):
    x = (x-x.min())/(x.max()-x.min())
    return x


if dataset == 'hanzi':
    data_dir = 'Your/Data/Dir/'
    if model_name == 'n2s':
        model = Noise2Same('trained_models/', 'Hanzi/N2S-Random', dim=2, in_channels=1)
    elif model_name == 'n2i':
        model = Noise2Info('trained_models/', 'Hanzi/N2I-Random', dim=2, in_channels=1)
    test = np.load(data_dir+'test.npy')[:, 3]
    test_gt = np.load(data_dir+'test.npy')[:, 0]
    preds = model.batch_predict(test.astype('float32'), batch_size=128)
    psnrs = [PSNR(norm(preds[idx]) * 255, test_gt[idx] * 255) for idx in range(len(test))]
    print(np.array(psnrs).mean())

elif dataset == 'bsd68':
    data_dir = 'Your/Data/Dir/'
    if model_name == 'n2s':
        model = Noise2Same('trained_models/', 'BSD68/N2S-Random', dim=2, in_channels=1)
    elif model_name == 'n2i':
        model = Noise2Info('trained_models/', 'BSD68/N2I-Random', dim=2, in_channels=1)
    groundtruth_data = np.load(data_dir+'bsd68_groundtruth.npy', allow_pickle=True)
    test_data = np.load(data_dir+'bsd68_gaussian25.npy', allow_pickle=True)
    preds = [model.predict(d.astype('float32')) for d in test_data]
    psnrs = [PSNR(preds[idx], groundtruth_data[idx]) for idx in range(len(test_data))]
    print(np.array(psnrs).mean())

elif dataset == 'imagenet':
    data_dir = 'Your/Data/Dir/'
    if model_name == 'n2s':
        model = Noise2Same('trained_models/', 'ImageNet/N2S-Donut', dim=2, in_channels=3)
    elif model_name == 'n2i':
        model = Noise2Info('trained_models/', 'ImageNet/N2I-Donut', dim=2, in_channels=3)
    path = data_dir+'test/'
    test_nlst = os.listdir(path)
    test = [np.load(path + test_n) for test_n in test_nlst]
    test_gt = [t[0] for t in test]
    test = [t[1] for t in test]
    preds = [model.predict(t.astype('float32'), im_mean=t.mean(axis=(0, 1)).astype('float32'),
                           im_std=t.std(axis=(0, 1)).astype('float32')) for t in test]
    psnrs = [PSNR(norm(p) * 255.0, gt) for p, gt in zip(preds, test_gt)]
    print(np.array(psnrs).mean())

elif dataset == 'sidd':
    data_dir = 'Your/Data/Dir/'
    if model_name == 'n2s':
        model = Noise2Same('trained_models/', 'SIDD/N2S-Random', dim=2, in_channels=3)
    elif model_name == 'n2i':
        model = Noise2Info('trained_models/', 'SIDD/N2I-Random', dim=2, in_channels=3)
    path = data_dir+'test/'
    test_nlst = os.listdir(path)
    test = [np.load(path + test_n).astype('float32') for test_n in test_nlst]
    test_gt = [t[0] for t in test]
    test = [t[1] for t in test]
    preds = [model.predict(t/255.0, im_mean=0.0, im_std=1.0) for t in test]
    psnrs = [PSNR(norm(p) * 255.0, norm(gt) * 255.0) for p, gt in zip(preds, test_gt)]
    print(np.array(psnrs).mean())
