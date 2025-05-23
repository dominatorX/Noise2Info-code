# Noise2Info
Official TensorFlow implementation for [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10378013)
"*Noise2Info: Noisy Image to Information of Noise for Self-Supervised Blind Image Denoising*".


## Environment Requirements
- jupyter
- python == 3.7.2
- tensorflow >=1.10 & <=1.15
- scipy
- skimage
- tifffile

## Usage

### To reproduce our results

#### Dataset and model checkpoint download
Benchmarks are widely compared and can be downloaded follow the link of another [paper](https://github.com/divelab/Noise2Same/tree/main/Denoising_data)
Checkpoints: [google drive](https://drive.google.com/drive/folders/13bWMm0q3spYPeiKLf4Lorxj2sXfa9xD0?usp=sharing)
### To train
We code the module for training in ``train.py``. After point its data_dir to your dataset and set dataset to train on, 
you can run
```angular2html
python3 train.py
```
for model training. 
### To test
Similar implementation can be found in ``test.py``. Set the checkpoint, you can run
```angular2html
python3 test.py
```
for model testing and reproducing our results. 
### To cite our paper
```angular2html
@inproceedings{wang2023noise2info,
  title={Noise2Info: Noisy image to information of noise for self-supervised image denoising},
  author={Wang, Jiachuan and Di, Shimin and Chen, Lei and Ng, Charles Wang Wai},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={16034--16043},
  year={2023}
}
```
