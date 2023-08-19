import logging, os, cv2
import numpy as np
from network_configure import conf_unet
from copy import deepcopy
import network_keras
from utils.predict_utils import get_coord, PercentileNormalizer, PadAndCropResizer
from utils.train_utils import augment_patch
from utils import train_utils
from network_configure import conf_basic_ops
import time
import tensorflow as tf
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

training_config = {'base_learning_rate': 0.0004,
                   'lr_decay_steps': 5000,
                   'lr_decay_rate': 0.5,
                   'lr_staircase': True}


class Noise2Info(object):
    def __init__(self, base_dir, name,
                 dim=2, in_channels=1, update_per_steps=1000, num_mc=10,
                 masking='gaussian', mask_perc=0.5,
                 opt_config=training_config, **kwargs):

        self.base_dir = base_dir  # model direction
        self.name = name  # model name
        self.dim = dim  # image dimension
        self.in_channels = in_channels  # image channels

        self.update_per_steps = update_per_steps  # update every k steps
        self.num_mc = num_mc

        self.masking = masking
        self.mask_perc = mask_perc

        self.opt_config = opt_config
        conf_unet['dimension'] = '%dD' % dim
        self.net = tf.keras.Sequential()
        self.net.add(network_keras.UNet(conf_unet))
        self.net.add(tf.keras.layers.BatchNormalization())
        self.net.add(tf.keras.layers.Activation('relu'))
        self.net.add(tf.keras.layers.Conv2D(self.in_channels, 1, 1, padding='same', use_bias=False,
                                              kernel_initializer=conf_basic_ops['kernel_initializer']) \
            if self.dim == 2 else tf.keras.layers.Conv3D(self.in_channels, 1, 1, padding='same', use_bias=False,
                                              kernel_initializer=conf_basic_ops['kernel_initializer']))

    def _input_fn(self, sources, patch_size, batch_size, mode='train'):
        # Stratified sampling inherited from Noise2Void: https://github.com/juglab/n2v
        get_stratified_coords = getattr(train_utils, 'get_stratified_coords%dD' % self.dim)
        rand_float_coords = getattr(train_utils, 'rand_float_coords%dD' % self.dim)

        def generator():
            while (True):
                source = sources[np.random.randint(len(sources))]
                valid_shape = source.shape[:-1] - np.array(patch_size)
                if any([s <= 0 for s in valid_shape]):
                    source_patch = augment_patch(source)
                else:
                    coords = [np.random.randint(0, shape_i + 1) for shape_i in valid_shape]
                    s = tuple([slice(coord, coord + size) for coord, size in zip(coords, patch_size)])
                    source_patch = augment_patch(source[s])

                mask = np.zeros_like(source_patch)
                for c in range(self.in_channels):
                    boxsize = np.round(np.sqrt(100 / self.mask_perc)).astype(np.int)
                    maskcoords = get_stratified_coords(rand_float_coords(boxsize),
                                                       box_size=boxsize, shape=tuple(patch_size))
                    indexing = maskcoords + (c,)
                    mask[indexing] = 1.0

                noise_patch = np.concatenate([np.random.normal(0, 0.2, source_patch.shape), mask], axis=-1)
                yield source_patch, noise_patch

        def generator_val():
            for idx in range(len(sources)):
                source_patch = sources[idx]
                patch_size = source_patch.shape[:-1]
                boxsize = np.round(np.sqrt(100 / self.mask_perc)).astype(np.int)
                maskcoords = get_stratified_coords(rand_float_coords(boxsize),
                                                   box_size=boxsize, shape=tuple(patch_size))
                indexing = maskcoords + (0,)
                mask = np.zeros_like(source_patch)
                mask[indexing] = 1.0
                noise_patch = np.concatenate([np.random.normal(0, 0.2, source_patch.shape), mask], axis=-1)
                yield source_patch, noise_patch

        output_types = (tf.float32, tf.float32)
        output_shapes = (tf.TensorShape(list(patch_size) + [self.in_channels]),
                         tf.TensorShape(list(patch_size) + [self.in_channels * 2]))
        gen = generator if mode != 'val' else generator_val
        dataset = tf.data.Dataset.from_generator(gen, output_types=output_types, output_shapes=output_shapes)
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    def train(self, source_lst, patch_size, validation=None, batch_size=64, steps=50000):
        assert len(patch_size) == self.dim
        assert len(source_lst[0].shape) == self.dim + 1
        assert source_lst[0].shape[-1] == self.in_channels

        num_rounds = steps//self.update_per_steps
        update_samples = self.update_per_steps // 10

        total_dims = self.in_channels
        for dim in patch_size:
            total_dims *= dim
        removed_noise = np.zeros((batch_size*update_samples, total_dims))

        writer = tf.summary.create_file_writer(self.base_dir + self.name)

        input_data = self._input_fn(source_lst, patch_size, batch_size=batch_size, mode='train')
        update_data = self._input_fn(source_lst, patch_size, batch_size=batch_size, mode='update')
        if validation is not None:
            val_data = self._input_fn(validation.astype('float32'), validation.shape[1:-1],
                                      batch_size=4, mode='val')

        model = network_keras.CustomModel(net=self.net, dimension=self.dim, in_channels=self.in_channels,
                                          masking=self.masking, num_mc=self.num_mc,
                                          removed_noise=removed_noise, writer=writer)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.opt_config['base_learning_rate'],
                                                                     self.opt_config['lr_decay_steps'],
                                                                     self.opt_config['lr_decay_rate'],
                                                                     self.opt_config['lr_staircase'])
        check_step = [0, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 50000]
        opt = tf.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(optimizer=opt, loss=lambda a,b: 0, metrics=['mse'])
        #for round in range(num_rounds):
        for round in range(len(check_step)-1):
            # update
            if round != 0:
                start = time.time()
                model.update(update_data, update_samples, batch_size, total_dims)
                print('update cost time', time.time() - start)

            # train
            start = time.time()
            if validation is not None:
                model.fit(input_data, batch_size=batch_size, epochs=1,
                          steps_per_epoch=check_step[round+1]-check_step[round], # self.update_per_steps - update_samples,
                          validation_data=val_data)
            else:
                model.fit(input_data, batch_size=batch_size, epochs=1,
                          steps_per_epoch=self.update_per_steps - update_samples)
            # print(opt._decayed_lr(tf.float32))
            print('train cost time', time.time() - start)
            model.net.save_weights(self.base_dir + self.name + '/'+str(round)+'.h5')
        model.net.save_weights(self.base_dir + self.name + '/last.h5')

    # Used for single image prediction
    def predict(self, image, resizer=PadAndCropResizer(), checkpoint_path=None, im_mean=None, im_std=None):

        im_mean, im_std = ((image.mean(), image.std()) if im_mean is None or im_std is None else (im_mean, im_std))
        image = (image - im_mean) / im_std

        if self.in_channels == 1:
            image = resizer.before(image, 2 ** (conf_unet['depth']), exclude=None)
            dataset = tf.data.Dataset.from_tensor_slices((image[None, ..., None], None)).batch(1)

            self.net.build(input_shape=(1, *image.shape, 1))
            if os.path.exists(self.base_dir + self.name + '/last.h5'):
                self.net.load_weights(self.base_dir + self.name + '/last.h5')
            elif checkpoint_path is not None:
                self.net.load_weights(checkpoint_path)
            else:
                print('no checkpoint exists')
                exit(1)

            for features, labels in dataset:
                image = self.net(features)[0, ..., 0]
            image = resizer.after(image, exclude=None)
        else:
            image = resizer.before(image, 2 ** (conf_unet['depth']), exclude=-1)

            self.net.build(input_shape=(1, *image.shape))
            if os.path.exists(self.base_dir + self.name + '/last.h5'):
                self.net.load_weights(self.base_dir + self.name + '/last.h5')
            elif checkpoint_path is not None:
                self.net.load_weights(checkpoint_path)
            else:
                print('no checkpoint exists')
                exit(1)

            dataset = tf.data.Dataset.from_tensor_slices((image[None, ...], None)).batch(1)
            for features, labels in dataset:
                image = self.net(features)[0, ...]
            image = resizer.after(image, exclude=-1)
        image = image * im_std + im_mean

        return image

    # Used for batch images prediction
    def batch_predict(self, images, resizer=PadAndCropResizer(), checkpoint_path=None,
                      im_mean=None, im_std=None, batch_size=32):

        im_mean, im_std = ((images.mean(), images.std()) if im_mean is None or im_std is None else (im_mean, im_std))

        images = (images - im_mean) / im_std
        images = resizer.before(images, 2 ** (conf_unet['depth']), exclude=0)

        self.net.build(input_shape=(batch_size, *images.shape[1:], 1))
        if os.path.exists(self.base_dir + self.name + '/last.h5'):
            self.net.load_weights(self.base_dir + self.name + '/last.h5')
        elif checkpoint_path is not None:
            self.net.load_weights(checkpoint_path)
        else:
            print('no checkpoint exists')
            exit(1)

        dataset = tf.data.Dataset.from_tensor_slices((images[..., None], None)).batch(batch_size)
        images = np.concatenate([self.net(features)[..., 0] for features, labels in dataset])
        images = resizer.after(images, exclude=0)
        images = images * im_std + im_mean

        return images

    # Used for extremely large input images
    def crop_predict(self, image, size, margin, resizer=PadAndCropResizer(), checkpoint_path=None,
                     im_mean=None, im_std=None):

        im_mean, im_std = ((image.mean(), image.std()) if im_mean is None or im_std is None else (im_mean, im_std))
        image = (image - im_mean) / im_std
        out_image = np.empty(image.shape, dtype='float32')
        for src_s, trg_s, mrg_s in get_coord(image.shape, size, margin):
            patch = resizer.before(image[src_s], 2 ** (self.net.depth), exclude=None)

            self.net.build(input_shape=(1, *patch.shape, 1))
            if os.path.exists(self.base_dir + self.name + '/last.h5'):
                self.net.load_weights(self.base_dir + self.name + '/last.h5')
            elif checkpoint_path is not None:
                self.net.load_weights(checkpoint_path)
            else:
                print('no checkpoint exists')
                exit(1)

            dataset = tf.data.Dataset.from_tensor_slices((patch[None, ..., None], None)).batch(1)
            patch = np.concatenate([self.net(features)[..., 0] for features, labels in dataset])
            patch = resizer.after(patch, exclude=None)
            out_image[trg_s] = patch[mrg_s]

        image = out_image * im_std + im_mean

        return image
