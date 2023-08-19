import logging, os, cv2
import numpy as np
from network_configure import conf_unet
from network import *
from copy import deepcopy
from utils.predict_utils import get_coord, PercentileNormalizer, PadAndCropResizer
from utils.train_utils import augment_patch
from utils import train_utils
from network_configure import conf_basic_ops
import time

# UNet implementation inherited from GVTNets: https://github.com/zhengyang-wang/GVTNets
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
        self.sgml = 1.  # lambda in loss fn
        self.masking = masking
        self.mask_perc = mask_perc

        self.update_per_steps = update_per_steps  # update every k steps
        self.num_mc = num_mc  # number of samples for Monte Carlo

        self.opt_config = opt_config
        conf_unet['dimension'] = '%dD' % dim
        self.net = UNet(conf_unet)

    def _model_fn(self, features, labels, mode):
        conv_op = convolution_2D if self.dim == 2 else convolution_3D
        axis = {3: [1, 2, 3, 4], 2: [1, 2, 3]}[self.dim]

        def image_summary(img):
            return tf.reduce_max(img, axis=1) if self.dim == 3 else img

        # Local average excluding the center pixel (donut)
        def mask_kernel(features):
            kernel = (np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]])
                      if self.dim == 2 else
                      np.array([[[0, 0.5, 0], [0.5, 1.0, 0.5], [0, 0.5, 0]],
                                [[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]],
                                [[0, 0.5, 0], [0.5, 1.0, 0.5], [0, 0.5, 0]]]))
            kernel = (kernel / kernel.sum())
            kernels = np.empty([3, 3, self.in_channels, self.in_channels])
            for i in range(self.in_channels):
                kernels[:, :, i, i] = kernel
            nn_conv_op = tf.nn.conv2d if self.dim == 2 else tf.nn.conv3d
            return nn_conv_op(features, tf.constant(kernels.astype('float32')),
                              [1] * self.dim + [1, 1], padding='SAME')

        if (not mode == tf.estimator.ModeKeys.PREDICT) or isinstance(features, dict):
            if isinstance(features, dict):
                features, noise, mask = features['features'], features['noise'], features['masks']
            else:
                noise, mask = tf.split(labels, [self.in_channels, self.in_channels], -1)

            if self.masking == 'gaussian':
                masked_features = (1 - mask) * features + mask * noise
            elif self.masking == 'donut':
                masked_features = (1 - mask) * features + mask * mask_kernel(features)
            else:
                raise NotImplementedError

            # Prediction from masked input
            with tf.variable_scope('main_unet', reuse=tf.compat.v1.AUTO_REUSE):
                out = self.net(masked_features, mode == tf.estimator.ModeKeys.TRAIN)
                out = batch_norm(out, mode == tf.estimator.ModeKeys.TRAIN, 'unet_out')
                out = relu(out)
                preds = conv_op(out, self.in_channels, 1, 1, False, name='out_conv')

            # Prediction from full input
            with tf.variable_scope('main_unet', reuse=tf.compat.v1.AUTO_REUSE):
                rawout = self.net(features, mode == tf.estimator.ModeKeys.TRAIN)
                rawout = batch_norm(rawout, mode == tf.estimator.ModeKeys.TRAIN, 'unet_out')
                rawout = relu(rawout)
                rawpreds = conv_op(rawout, self.in_channels, 1, 1, False, name='out_conv')

            # Loss components
            n_mask = tf.reduce_sum(mask)
            rec_mse = tf.reduce_mean(tf.square(rawpreds - features), axis=None)
            inv_mse = tf.reduce_sum(tf.square(rawpreds - preds) * mask) / n_mask
            bsp_mse = tf.reduce_sum(tf.square(features - preds) * mask) / n_mask

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    'rec': rec_mse, 'inv': inv_mse * n_mask,
                    'n_mask': n_mask, 'removed': rawpreds - features
                }
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
            # Tensorboard display
            tf.summary.image('1_inputs', image_summary(features), max_outputs=3)
            tf.summary.image('2_raw_predictions', image_summary(rawpreds), max_outputs=3)
            tf.summary.image('3_mask', image_summary(mask), max_outputs=3)
            tf.summary.image('4_masked_predictions', image_summary(preds), max_outputs=3)
            tf.summary.image('5_difference', image_summary(rawpreds - preds), max_outputs=3)
            tf.summary.image('6_rec_error', image_summary(preds - features), max_outputs=3)
            tf.summary.scalar('reconstruction', rec_mse, family='loss_metric')
            tf.summary.scalar('invariance', inv_mse, family='loss_metric')
            tf.summary.scalar('blind_spot', bsp_mse, family='loss_metric')

        else:
            with tf.variable_scope('main_unet'):
                out = self.net(features, mode == tf.estimator.ModeKeys.TRAIN)
                out = batch_norm(out, mode == tf.estimator.ModeKeys.TRAIN, 'unet_out')
                out = relu(out)
                preds = conv_op(out, self.in_channels, 1, 1, False, name='out_conv')
            return tf.estimator.EstimatorSpec(mode=mode, predictions=preds)

        loss = rec_mse + 2 * self.sgml * tf.sqrt(inv_mse)

        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_or_create_global_step()
            learning_rate = tf.train.exponential_decay(self.opt_config['base_learning_rate'],
                                                       global_step,
                                                       self.opt_config['lr_decay_steps'],
                                                       self.opt_config['lr_decay_rate'],
                                                       self.opt_config['lr_staircase'])
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='main_unet')
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step)
        else:
            train_op = None

        metrics = {'loss_metric/invariance': tf.metrics.mean(inv_mse),
                   'loss_metric/blind_spot': tf.metrics.mean(bsp_mse),
                   'loss_metric/reconstruction': tf.metrics.mean(rec_mse)}

        return tf.estimator.EstimatorSpec(mode=mode, predictions=preds, loss=loss, train_op=train_op,
                                          eval_metric_ops=metrics)

    def _input_fn(self, sources, patch_size, batch_size, is_train=True):
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
        gen = generator if is_train else generator_val
        dataset = tf.data.Dataset.from_generator(gen, output_types=output_types, output_shapes=output_shapes)
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    def _update_input_fn(self, sources, patch_size, batch_size, size):
        # Stratified sampling inherited from Noise2Void: https://github.com/juglab/n2v
        get_stratified_coords = getattr(train_utils, 'get_stratified_coords%dD' % self.dim)
        rand_float_coords = getattr(train_utils, 'rand_float_coords%dD' % self.dim)

        noise, masks, features = [], [], []
        for i in range(size*batch_size):
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
            noise.append(np.random.normal(0, 0.2, source_patch.shape))
            masks.append(mask)
            features.append(source_patch)
        noise = np.stack(noise).astype('float32')
        masks = np.stack(masks).astype('float32')
        features = np.stack(features).astype('float32')
        update_fn = tf.estimator.inputs.numpy_input_fn(x={"features":features, "masks":masks, "noise":noise},
                                                       num_epochs=1, shuffle=False, batch_size=batch_size)
        return update_fn

    def update_sgml(self, E_fx_x, E_f_xj_fx_j, sgm_r=0):
        sgml_ = (E_f_xj_fx_j)**0.5 + (E_f_xj_fx_j + E_fx_x-sgm_r)**0.5
        print('new sigma_loss is ', sgml_)
        if sgml_ < self.sgml:
            self.sgml = float(sgml_)
            print('sigma_loss updated to ', self.sgml)

    def train(self, source_lst, patch_size, validation=None, batch_size=64, save_steps=1000, log_steps=200,
              steps=50000):
        assert len(patch_size) == self.dim
        assert len(source_lst[0].shape) == self.dim + 1
        assert source_lst[0].shape[-1] == self.in_channels


        num_rounds = steps//self.update_per_steps
        update_samples = self.update_per_steps // 10

        self.batch_size = batch_size
        total_dims = self.in_channels
        for dim in patch_size:
            total_dims *= dim
        removed_noise = np.zeros((batch_size * update_samples, total_dims))

        ses_config = tf.ConfigProto()
        ses_config.gpu_options.allow_growth = True

        run_config = tf.estimator.RunConfig(model_dir=self.base_dir + '/' + self.name,
                                            save_checkpoints_steps=save_steps,
                                            session_config=ses_config,
                                            log_step_count_steps=log_steps,
                                            save_summary_steps=log_steps,
                                            keep_checkpoint_max=2)

        estimator = tf.estimator.Estimator(model_fn=self._model_fn,
                                           model_dir=self.base_dir + '/' + self.name,
                                           config=run_config)

        input_fn = lambda: self._input_fn(source_lst, patch_size, batch_size=batch_size)
        for round in range(num_rounds):
            # update
            if round != 0:
                start = time.time()
                update_fn = self._update_input_fn(deepcopy(source_lst), patch_size, batch_size=batch_size,
                                                  size=update_samples)
                step, rec, inv, n_mask, sgm_r = 0, 0, 0, 0, 0
                for update_vars in estimator.predict(input_fn=update_fn, yield_single_examples=False):
                    step += 1
                    rec += update_vars['rec']
                    inv += update_vars['inv']
                    n_mask += update_vars['n_mask']
                    removed_noise[(step - 1) * batch_size: step * batch_size] = \
                        np.sort((update_vars['removed']).reshape((self.batch_size, -1)))
                for _ in range(self.num_mc):
                    samples = np.sort(np.random.choice(removed_noise.reshape(update_samples * batch_size * total_dims),
                                                       [1, total_dims]))
                    sgm_r += np.square(samples - removed_noise).mean((0, 1))
                self.update_sgml(rec / update_samples, inv / n_mask, sgm_r=sgm_r / self.num_mc)
                print('updation time', time.time() - start)

            # train
            start = time.time()
            if validation is not None:
                train_spec = tf.estimator.TrainSpec(input_fn=input_fn,
                                                    max_steps=(round+1)*(self.update_per_steps-update_samples))
                val_input_fn = lambda: self._input_fn(validation.astype('float32'),
                                                      validation.shape[1:-1],
                                                      batch_size=4,
                                                      is_train=False)
                eval_spec = tf.estimator.EvalSpec(input_fn=val_input_fn, throttle_secs=120)
                tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
            else:
                estimator.train(input_fn=input_fn, steps=self.update_per_steps-update_samples)

            print('training time', time.time()-start)

    # Used for single image prediction
    def predict(self, image, resizer=PadAndCropResizer(), checkpoint_path=None,
                im_mean=None, im_std=None):

        tf.logging.set_verbosity(tf.logging.ERROR)
        estimator = tf.estimator.Estimator(model_fn=self._model_fn,
                                           model_dir=self.base_dir + '/' + self.name)

        im_mean, im_std = ((image.mean(), image.std()) if im_mean is None or im_std is None else (im_mean, im_std))
        image = (image - im_mean) / im_std
        if self.in_channels == 1:
            image = resizer.before(image, 2 ** (self.net.depth), exclude=None)
            input_fn = tf.estimator.inputs.numpy_input_fn(x=image[None, ..., None], batch_size=1, num_epochs=1,
                                                          shuffle=False)
            image = list(estimator.predict(input_fn=input_fn, checkpoint_path=checkpoint_path))[0][..., 0]
            image = resizer.after(image, exclude=None)
        else:
            image = resizer.before(image, 2 ** (self.net.depth), exclude=-1)
            input_fn = tf.estimator.inputs.numpy_input_fn(x=image[None], batch_size=1, num_epochs=1, shuffle=False)
            image = list(estimator.predict(input_fn=input_fn, checkpoint_path=checkpoint_path))[0]
            image = resizer.after(image, exclude=-1)
        image = image * im_std + im_mean

        return image

    # Used for batch images prediction
    def batch_predict(self, images, resizer=PadAndCropResizer(), checkpoint_path=None,
                      im_mean=None, im_std=None, batch_size=32):

        tf.logging.set_verbosity(tf.logging.ERROR)
        estimator = tf.estimator.Estimator(model_fn=self._model_fn,
                                           model_dir=self.base_dir + '/' + self.name)

        im_mean, im_std = ((images.mean(), images.std()) if im_mean is None or im_std is None else (im_mean, im_std))

        images = (images - im_mean) / im_std
        images = resizer.before(images, 2 ** (self.net.depth), exclude=0)
        input_fn = tf.estimator.inputs.numpy_input_fn(x=images[..., None], batch_size=batch_size, num_epochs=1,
                                                      shuffle=False)
        images = np.stack(list(estimator.predict(input_fn=input_fn, checkpoint_path=checkpoint_path)))[..., 0]
        images = resizer.after(images, exclude=0)
        images = images * im_std + im_mean

        return images

    # Used for extremely large input images
    def crop_predict(self, image, size, margin, resizer=PadAndCropResizer(), checkpoint_path=None,
                     im_mean=None, im_std=None):

        tf.logging.set_verbosity(tf.logging.ERROR)
        estimator = tf.estimator.Estimator(model_fn=self._model_fn,
                                           model_dir=self.base_dir + '/' + self.name)

        im_mean, im_std = ((image.mean(), image.std()) if im_mean is None or im_std is None else (im_mean, im_std))
        image = (image - im_mean) / im_std
        out_image = np.empty(image.shape, dtype='float32')
        for src_s, trg_s, mrg_s in get_coord(image.shape, size, margin):
            patch = resizer.before(image[src_s], 2 ** (self.net.depth), exclude=None)
            input_fn = tf.estimator.inputs.numpy_input_fn(x=patch[None, ..., None], batch_size=1, num_epochs=1,
                                                          shuffle=False)
            patch = list(estimator.predict(input_fn=input_fn, checkpoint_path=checkpoint_path))[0][..., 0]
            patch = resizer.after(patch, exclude=None)
            out_image[trg_s] = patch[mrg_s]

        image = out_image * im_std + im_mean

        return image

class Noise2Same(object):

    def __init__(self, base_dir, name, 
                 dim=2, in_channels=1, lmbd=None, 
                 masking='gaussian', mask_perc=0.5,
                 opt_config=training_config, **kwargs):

        self.base_dir = base_dir # model direction
        self.name = name # model name
        self.dim = dim # image dimension
        self.in_channels = in_channels # image channels
        self.lmbd = lmbd # lambda in loss fn
        self.masking = masking
        self.mask_perc = mask_perc
        
        self.opt_config = opt_config
        conf_unet['dimension'] = '%dD'%dim
        self.net = UNet(conf_unet)
        
    def _model_fn(self, features, labels, mode):
        conv_op = convolution_2D if self.dim==2 else convolution_3D
        axis = {3:[1,2,3,4], 2:[1,2,3]}[self.dim]
        
        def image_summary(img):
            return tf.reduce_max(img, axis=1) if self.dim == 3 else img
        
        # Local average excluding the center pixel (donut)
        def mask_kernel(features):
            kernel = (np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]]) 
                      if self.dim == 2 else 
                      np.array([[[0, 0.5, 0], [0.5, 1.0, 0.5], [0, 0.5, 0]],
                                [[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]],
                                [[0, 0.5, 0], [0.5, 1.0, 0.5], [0, 0.5, 0]]]))
            kernel = (kernel/kernel.sum())
            kernels = np.empty([3, 3, self.in_channels, self.in_channels])
            for i in range(self.in_channels):
                kernels[:,:,i,i] = kernel
            nn_conv_op = tf.nn.conv2d if self.dim == 2 else tf.nn.conv3d
            return nn_conv_op(features, tf.constant(kernels.astype('float32')), 
                              [1]*self.dim+[1,1], padding='SAME')
        
        if not mode == tf.estimator.ModeKeys.PREDICT:
            noise, mask = tf.split(labels, [self.in_channels, self.in_channels], -1)
            
            if self.masking == 'gaussian':
                masked_features = (1 - mask) * features + mask * noise
            elif self.masking == 'donut':
                masked_features = (1 - mask) * features + mask * mask_kernel(features)
            else:
                raise NotImplementedError
            
            # Prediction from masked input
            with tf.variable_scope('main_unet', reuse=tf.compat.v1.AUTO_REUSE):
                out = self.net(masked_features, mode == tf.estimator.ModeKeys.TRAIN)
                out = batch_norm(out, mode == tf.estimator.ModeKeys.TRAIN, 'unet_out')
                out = relu(out)
                preds = conv_op(out, self.in_channels, 1, 1, False, name = 'out_conv')
                
            # Prediction from full input
            with tf.variable_scope('main_unet', reuse=tf.compat.v1.AUTO_REUSE):
                rawout = self.net(features, mode == tf.estimator.ModeKeys.TRAIN)
                rawout = batch_norm(rawout, mode == tf.estimator.ModeKeys.TRAIN, 'unet_out')
                rawout = relu(rawout)
                rawpreds = conv_op(rawout, self.in_channels, 1, 1, False, name = 'out_conv')
            
            # Loss components
            rec_mse = tf.reduce_mean(tf.square(rawpreds - features), axis=None)
            inv_mse = tf.reduce_sum(tf.square(rawpreds - preds) * mask) / tf.reduce_sum(mask)
            bsp_mse = tf.reduce_sum(tf.square(features - preds) * mask) / tf.reduce_sum(mask)

            # Tensorboard display
            tf.summary.image('1_inputs', image_summary(features), max_outputs=3)
            tf.summary.image('2_raw_predictions', image_summary(rawpreds), max_outputs=3)
            tf.summary.image('3_mask', image_summary(mask), max_outputs=3)
            tf.summary.image('4_masked_predictions', image_summary(preds), max_outputs=3)
            tf.summary.image('5_difference', image_summary(rawpreds-preds), max_outputs=3)
            tf.summary.image('6_rec_error', image_summary(preds-features), max_outputs=3)
            tf.summary.scalar('reconstruction', rec_mse, family='loss_metric') 
            tf.summary.scalar('invariance', inv_mse, family='loss_metric') 
            tf.summary.scalar('blind_spot', bsp_mse, family='loss_metric')
                
        else:
            with tf.variable_scope('main_unet'):
                out = self.net(features, mode == tf.estimator.ModeKeys.TRAIN)
                out = batch_norm(out, mode == tf.estimator.ModeKeys.TRAIN, 'unet_out')
                out = relu(out)
                preds = conv_op(out, self.in_channels, 1, 1, False, name = 'out_conv')
            return tf.estimator.EstimatorSpec(mode=mode, predictions=preds)
        
        lmbd = 2 if self.lmbd is None else self.lmbd
        loss = rec_mse + lmbd*tf.sqrt(inv_mse)

        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_or_create_global_step()
            learning_rate = tf.train.exponential_decay(self.opt_config['base_learning_rate'], 
                                                       global_step, 
                                                       self.opt_config['lr_decay_steps'], 
                                                       self.opt_config['lr_decay_rate'], 
                                                       self.opt_config['lr_staircase'])
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='main_unet')
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step)
        else:
            train_op = None
        
        metrics = {'loss_metric/invariance':tf.metrics.mean(inv_mse),
                              'loss_metric/blind_spot':tf.metrics.mean(bsp_mse), 
                              'loss_metric/reconstruction':tf.metrics.mean(rec_mse)}
        
        return tf.estimator.EstimatorSpec(mode=mode, predictions=preds, loss=loss, train_op=train_op, 
                                          eval_metric_ops=metrics)


    def _input_fn(self, sources, patch_size, batch_size, is_train=True):
        # Stratified sampling inherited from Noise2Void: https://github.com/juglab/n2v
        get_stratified_coords = getattr(train_utils, 'get_stratified_coords%dD'%self.dim)
        rand_float_coords = getattr(train_utils, 'rand_float_coords%dD'%self.dim)
        
        def generator():
            while(True):
                source = sources[np.random.randint(len(sources))]
                valid_shape = source.shape[:-1] - np.array(patch_size)
                if any([s<=0 for s in valid_shape]):
                    source_patch = augment_patch(source)
                else:
                    coords = [np.random.randint(0, shape_i+1) for shape_i in valid_shape]
                    s = tuple([slice(coord, coord+size) for coord, size in zip(coords, patch_size)])
                    source_patch = augment_patch(source[s])
                
                mask = np.zeros_like(source_patch)
                for c in range(self.in_channels):
                    boxsize = np.round(np.sqrt(100/self.mask_perc)).astype(np.int)
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
                boxsize = np.round(np.sqrt(100/self.mask_perc)).astype(np.int)
                maskcoords = get_stratified_coords(rand_float_coords(boxsize), 
                                                   box_size=boxsize, shape=tuple(patch_size))
                indexing = maskcoords + (0,)
                mask = np.zeros_like(source_patch)
                mask[indexing] = 1.0
                noise_patch = np.concatenate([np.random.normal(0, 0.2, source_patch.shape), mask], axis=-1)
                yield source_patch, noise_patch

        output_types = (tf.float32, tf.float32)
        output_shapes = (tf.TensorShape(list(patch_size) + [self.in_channels]), 
                                             tf.TensorShape(list(patch_size) + [self.in_channels*2]))
        gen = generator if is_train else generator_val
        dataset = tf.data.Dataset.from_generator(gen, output_types=output_types, output_shapes=output_shapes)
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return dataset


    def train(self, source_lst, patch_size, validation=None, batch_size=64, save_steps=1000, log_steps=200, steps=50000):
        assert len(patch_size)==self.dim
        assert len(source_lst[0].shape)==self.dim + 1
        assert source_lst[0].shape[-1]==self.in_channels

        ses_config = tf.ConfigProto()
        ses_config.gpu_options.allow_growth = True

        run_config = tf.estimator.RunConfig(model_dir=self.base_dir+'/'+self.name, 
                                            save_checkpoints_steps=save_steps,
                                            session_config=ses_config, 
                                            log_step_count_steps=log_steps,
                                            save_summary_steps=log_steps,
                                            keep_checkpoint_max=2)

        estimator = tf.estimator.Estimator(model_fn=self._model_fn, 
                                             model_dir=self.base_dir+'/'+self.name, 
                                             config=run_config)
        
        input_fn = lambda: self._input_fn(source_lst, patch_size, batch_size=batch_size)
        
        if validation is not None:
            train_spec = tf.estimator.TrainSpec(input_fn=input_fn, max_steps=steps)
            val_input_fn = lambda: self._input_fn(validation.astype('float32'), 
                                                  validation.shape[1:-1], 
                                                  batch_size=4, 
                                                  is_train=False)
            eval_spec = tf.estimator.EvalSpec(input_fn=val_input_fn, throttle_secs=120)
            tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        else:
            estimator.train(input_fn=input_fn, steps=steps)
            

    # Used for single image prediction
    def predict(self, image, resizer=PadAndCropResizer(), checkpoint_path=None,
               im_mean=None, im_std=None):

        tf.logging.set_verbosity(tf.logging.ERROR)
        estimator = tf.estimator.Estimator(model_fn=self._model_fn, 
                                           model_dir=self.base_dir+'/'+self.name)
        
        im_mean, im_std = ((image.mean(), image.std()) if im_mean is None or im_std is None else (im_mean, im_std)) 
        image = (image - im_mean)/im_std
        if self.in_channels == 1:
            image = resizer.before(image, 2 ** (self.net.depth), exclude=None)
            input_fn = tf.estimator.inputs.numpy_input_fn(x=image[None, ..., None], batch_size=1, num_epochs=1, shuffle=False)
            image = list(estimator.predict(input_fn=input_fn, checkpoint_path=checkpoint_path))[0][..., 0]
            image = resizer.after(image, exclude=None)
        else:
            image = resizer.before(image, 2 ** (self.net.depth), exclude=-1)
            input_fn = tf.estimator.inputs.numpy_input_fn(x=image[None], batch_size=1, num_epochs=1, shuffle=False)
            image = list(estimator.predict(input_fn=input_fn, checkpoint_path=checkpoint_path))[0]
            image = resizer.after(image, exclude=-1)
        image = image*im_std + im_mean

        return image
    
    # Used for batch images prediction
    def batch_predict(self, images, resizer=PadAndCropResizer(), checkpoint_path=None,
               im_mean=None, im_std=None, batch_size=32):

        tf.logging.set_verbosity(tf.logging.ERROR)
        estimator = tf.estimator.Estimator(model_fn=self._model_fn, 
                                            model_dir=self.base_dir+'/'+self.name)
        
        im_mean, im_std = ((images.mean(), images.std()) if im_mean is None or im_std is None else (im_mean, im_std)) 
        
        images = (images - im_mean)/im_std
        images = resizer.before(images, 2 ** (self.net.depth), exclude=0)
        input_fn = tf.estimator.inputs.numpy_input_fn(x=images[ ..., None], batch_size=batch_size, num_epochs=1, shuffle=False)
        images = np.stack(list(estimator.predict(input_fn=input_fn, checkpoint_path=checkpoint_path)))[..., 0]
        images = resizer.after(images, exclude=0)
        images = images*im_std + im_mean

        return images

    # Used for extremely large input images
    def crop_predict(self, image, size, margin, resizer=PadAndCropResizer(), checkpoint_path=None,
               im_mean=None, im_std=None):

        tf.logging.set_verbosity(tf.logging.ERROR)
        estimator = tf.estimator.Estimator(model_fn=self._model_fn, 
                                            model_dir=self.base_dir+'/'+self.name)
        
        im_mean, im_std = ((image.mean(), image.std()) if im_mean is None or im_std is None else (im_mean, im_std)) 
        image = (image - im_mean)/im_std
        out_image = np.empty(image.shape, dtype='float32')
        for src_s, trg_s, mrg_s in get_coord(image.shape, size, margin):
            patch = resizer.before(image[src_s], 2 ** (self.net.depth), exclude=None)
            input_fn = tf.estimator.inputs.numpy_input_fn(x=patch[None, ..., None], batch_size=1, num_epochs=1, shuffle=False)
            patch = list(estimator.predict(input_fn=input_fn, checkpoint_path=checkpoint_path))[0][..., 0]
            patch = resizer.after(patch, exclude=None)
            out_image[trg_s] = patch[mrg_s]
            
        image = out_image*im_std + im_mean

        return image

'''
class Noise2Info1(object):
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

        opt = tf.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(optimizer=opt, loss=lambda a,b: 0, metrics=['mse'])
        for round in range(num_rounds):
            start = time.time()
            model.update(update_data, update_samples, batch_size, total_dims)
            mid = time.time()
            print('update cost time', mid - start)
            if validation is not None:
                model.fit(input_data, batch_size=batch_size, epochs=1,
                          steps_per_epoch=self.update_per_steps - update_samples,
                          validation_data=val_data)
            else:
                model.fit(input_data, batch_size=batch_size, epochs=1,
                          steps_per_epoch=self.update_per_steps - update_samples)
            # print(opt._decayed_lr(tf.float32))
            print('train cost time', time.time() - mid)
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


class Noise2Info2(object):
    def __init__(self, base_dir, name,
                 dim=2, in_channels=1, update_per_steps=1000, num_mc=10,
                 masking='gaussian', mask_perc=0.5,
                 opt_config=training_config, **kwargs):

        self.base_dir = base_dir  # model direction
        self.name = name  # model name
        self.dim = dim  # image dimension
        self.in_channels = in_channels  # image channels

        self.sgml = 1 # sigma_loss
        self.update_per_steps = update_per_steps  # update every k steps
        self.num_mc = num_mc
        self.removed_noise = None

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

    def update_sgml(self, E_fx_x, E_f_xj_fx_j, sgm_r=0):
        sgml_ = (E_f_xj_fx_j)**0.5 + (E_f_xj_fx_j + E_fx_x-sgm_r)**0.5
        print('new sigma_loss is ', sgml_)
        if sgml_ < self.sgml:
            self.sgml = float(sgml_)
            print('sigma_loss updated to ', self.sgml)

    @tf.function
    def model(self, features, labels, mode='train', step=0):
        # conv_op = convolution_2D if self.dim == 2 else convolution_3D
        axis = {3: [1, 2, 3, 4], 2: [1, 2, 3]}[self.dim]

        def image_summary(img):
            return tf.reduce_max(img, axis=1) if self.dim == 3 else img

        # Local average excluding the center pixel (donut)
        def mask_kernel(features):
            kernel = (np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]])
                      if self.dim == 2 else
                      np.array([[[0, 0.5, 0], [0.5, 1.0, 0.5], [0, 0.5, 0]],
                                [[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]],
                                [[0, 0.5, 0], [0.5, 1.0, 0.5], [0, 0.5, 0]]]))
            kernel = (kernel / kernel.sum())
            if self.dim == 2:
                kernels = np.empty([3, 3, self.in_channels, self.in_channels])
                for i in range(self.in_channels):
                    kernels[:, :, i, i] = kernel
            else:
                kernels = np.empty([3, 3, 3, self.in_channels, self.in_channels])
                for i in range(self.in_channels):
                    kernels[:, :, :, i, i] = kernel
            nn_conv_op = tf.nn.conv2d if self.dim == 2 else tf.nn.conv3d
            return nn_conv_op(features, tf.constant(kernels.astype('float32')),
                              [1] * self.dim + [1, 1], padding='SAME')

        if not mode == 'predict':
            if mode == 'train':
                with tf.GradientTape() as tape:
                    noise, mask = tf.split(labels, [self.in_channels, self.in_channels], -1)

                    if self.masking == 'gaussian':
                        masked_features = (1 - mask) * features + mask * noise
                    elif self.masking == 'donut':
                        masked_features = (1 - mask) * features + mask * mask_kernel(features)
                    else:
                        raise NotImplementedError

                    preds = self.net(masked_features, training=mode=='train')
                    rawpreds = self.net(features, training=mode=='train')

                    # Loss components
                    n_mask = tf.reduce_sum(mask)
                    rec_mse = tf.reduce_mean(tf.square(rawpreds - features), axis=None)
                    inv_mse = tf.reduce_sum(tf.square(rawpreds - preds) * mask) / n_mask
                    bsp_mse = tf.reduce_sum(tf.square(features - preds) * mask) / n_mask
                    loss = rec_mse + 2 * self.sgml * tf.sqrt(inv_mse)
            else:
                noise, mask = tf.split(labels, [self.in_channels, self.in_channels], -1)

                if self.masking == 'gaussian':
                    masked_features = (1 - mask) * features + mask * noise
                elif self.masking == 'donut':
                    masked_features = (1 - mask) * features + mask * mask_kernel(features)
                else:
                    raise NotImplementedError

                preds = self.net(masked_features, training=mode == 'train')
                rawpreds = self.net(features, training=mode == 'train')

                # Loss components
                n_mask = tf.reduce_sum(mask)
                rec_mse = tf.reduce_mean(tf.square(rawpreds - features), axis=None)
                inv_mse = tf.reduce_sum(tf.square(rawpreds - preds) * mask) / n_mask
                bsp_mse = tf.reduce_sum(tf.square(features - preds) * mask) / n_mask
                loss = rec_mse + 2 * self.sgml * tf.sqrt(inv_mse)
            if mode == 'update':
                return rec_mse.numpy(), inv_mse.numpy()*n_mask.numpy(), n_mask.numpy(), \
                       np.sort((rawpreds - features).numpy().reshape((self.batch_size, -1)))

            if mode == 'train':
                learning_rate = tf.compat.v1.train.exponential_decay(self.opt_config['base_learning_rate'], step,
                                                           self.opt_config['lr_decay_steps'],
                                                           self.opt_config['lr_decay_rate'],
                                                           self.opt_config['lr_staircase'])

                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
                grads = tape.gradient(loss, self.net.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.net.trainable_variables))
                return
            elif mode=='eval':
                with self.writer.as_default():
                    tf.summary.image('1_inputs', image_summary(features), max_outputs=3, step=step)
                    tf.summary.image('2_raw_predictions', image_summary(rawpreds), max_outputs=3, step=step)
                    tf.summary.image('3_mask', image_summary(mask), max_outputs=3, step=step)
                    tf.summary.image('4_masked_predictions', image_summary(preds), max_outputs=3, step=step)
                    tf.summary.image('5_difference', image_summary(rawpreds - preds), max_outputs=3, step=step)
                    tf.summary.image('6_rec_error', image_summary(preds - features), max_outputs=3, step=step)
                    tf.summary.scalar('loss_metric/reconstruction', rec_mse, step=step)
                    tf.summary.scalar('loss_metric/invariance', inv_mse, step=step)
                    tf.summary.scalar('loss_metric/blind_spot', bsp_mse, step=step)
                    self.writer.flush()
                return {'loss_metric/invariance': tf.metrics.mean(inv_mse),
                           'loss_metric/blind_spot': tf.metrics.mean(bsp_mse),
                           'loss_metric/reconstruction': tf.metrics.mean(rec_mse)}
        else:
            preds = self.net(features)
            return preds.numpy()

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

    def update(self, update_data, update_samples, batch_size, total_dims):
        step, rec, inv, n_mask, sgm_r = 0, 0, 0, 0, 0
        for features, labels in update_data:
            if step == update_samples: break
            step += 1
            rec_, inv_, n_mask_, removed_noise = self.model(features, labels, 'update')
            rec += rec_
            inv += inv_
            n_mask += n_mask_
            self.removed_noise[(step - 1) * batch_size: step * batch_size] = removed_noise
        for _ in range(self.num_mc):
            samples = np.sort(np.random.choice(self.removed_noise.reshape(update_samples * batch_size * total_dims),
                                               [1, total_dims]))
            sgm_r += np.square(samples - self.removed_noise).mean((0, 1))
        self.update_sgml(rec / update_samples, inv / n_mask, sgm_r=sgm_r / self.num_mc)

    def train(self, source_lst, patch_size, validation=None, batch_size=64, save_steps=1000, log_steps=200,
              steps=50000):
        assert len(patch_size) == self.dim
        assert len(source_lst[0].shape) == self.dim + 1
        assert source_lst[0].shape[-1] == self.in_channels

        num_rounds = steps//self.update_per_steps
        update_samples = self.update_per_steps // 10

        self.batch_size = batch_size
        total_dims = self.in_channels
        for dim in patch_size:
            total_dims *= dim
        self.removed_noise = np.zeros((batch_size*update_samples, total_dims))

        #ses_config = tf.ConfigProto()
        #ses_config.gpu_options.allow_growth = True

        self.writer = tf.summary.create_file_writer(self.base_dir + '/' + self.name)

        input_data = self._input_fn(source_lst, patch_size, batch_size=batch_size, mode='train')
        update_data = self._input_fn(source_lst, patch_size, batch_size=batch_size, mode='update')
        if validation is not None:
            val_data = self._input_fn(validation.astype('float32'), validation.shape[1:-1],
                                      batch_size=4, mode='val')
        step = 0
        for features, labels in input_data:
            # update
            if step % (self.update_per_steps - update_samples) == 0:
                if step != 0:
                    print('train cost time', time.time() - mid)
                start = time.time()
                self.update(update_data, update_samples, batch_size, total_dims)
                mid = time.time()
                print('update cost time', mid-start)
            # train
            self.model(features, labels, 'train', step)
            # save
            if (step+1) % save_steps == 0:
                self.net.save_weights(self.base_dir + '/' + self.name+'/'+str(step+1))
            # evaluate
            if (validation is not None) and (step+1) % log_steps == 0:
                for val_features, val_labels in val_data:
                    self.model(val_features, val_labels, 'val', step + 1)
            # end
            if (step+1) == num_rounds*(self.update_per_steps - update_samples):
                break
            step += 1
        print('train cost time', time.time() - mid)
        self.net.save_weights(self.base_dir + '/' + self.name + '/last')
    # Used for single image prediction
    def predict(self, image, resizer=PadAndCropResizer(), checkpoint_path=None, im_mean=None, im_std=None):

        im_mean, im_std = ((image.mean(), image.std()) if im_mean is None or im_std is None else (im_mean, im_std))
        image = (image - im_mean) / im_std
        self.net.load_weights(checkpoint_path if checkpoint_path is not None else
                              self.base_dir + '/' + self.name + '/last')
        if self.in_channels == 1:
            image = resizer.before(image, 2 ** (conf_unet['depth']), exclude=None)
            dataset = tf.data.Dataset.from_tensor_slices((image[None, ..., None], None)).batch(1)
            for features, labels in dataset:
                image = self.model(features, labels, 'predict')[0, ..., 0]
            image = resizer.after(image, exclude=None)
        else:
            image = resizer.before(image, 2 ** (conf_unet['depth']), exclude=-1)
            dataset = tf.data.Dataset.from_tensor_slices((image[None, ...], None)).batch(1)
            for features, labels in dataset:
                image = self.model(features, labels, 'predict')[0, ...]
            # image = list(estimator.predict(input_fn=input_fn, checkpoint_path=checkpoint_path))[0]
            image = resizer.after(image, exclude=-1)
        image = image * im_std + im_mean

        return image

    # Used for batch images prediction
    def batch_predict(self, images, resizer=PadAndCropResizer(), checkpoint_path=None,
                      im_mean=None, im_std=None, batch_size=32):
        self.net.load_weights(checkpoint_path if checkpoint_path is not None else
                              self.base_dir + '/' + self.name + '/last')

        im_mean, im_std = ((images.mean(), images.std()) if im_mean is None or im_std is None else (im_mean, im_std))

        images = (images - im_mean) / im_std
        images = resizer.before(images, 2 ** (conf_unet['depth']), exclude=0)

        dataset = tf.data.Dataset.from_tensor_slices((images[..., None], None)).batch(batch_size)

        images = np.stack([self.model(features, labels, 'predict')[0, ..., 0] for features, labels in dataset])
        images = resizer.after(images, exclude=0)
        images = images * im_std + im_mean

        return images

    # Used for extremely large input images
    def crop_predict(self, image, size, margin, resizer=PadAndCropResizer(), checkpoint_path=None,
                     im_mean=None, im_std=None):

        tf.logging.set_verbosity(tf.logging.ERROR)
        estimator = tf.estimator.Estimator(model_fn=self._model_fn,
                                           model_dir=self.base_dir + '/' + self.name)

        im_mean, im_std = ((image.mean(), image.std()) if im_mean is None or im_std is None else (im_mean, im_std))
        image = (image - im_mean) / im_std
        out_image = np.empty(image.shape, dtype='float32')
        for src_s, trg_s, mrg_s in get_coord(image.shape, size, margin):
            patch = resizer.before(image[src_s], 2 ** (self.net.depth), exclude=None)
            input_fn = tf.estimator.inputs.numpy_input_fn(x=patch[None, ..., None], batch_size=1, num_epochs=1,
                                                          shuffle=False)
            patch = list(estimator.predict(input_fn=input_fn, checkpoint_path=checkpoint_path))[0][..., 0]
            patch = resizer.after(patch, exclude=None)
            out_image[trg_s] = patch[mrg_s]

        image = out_image * im_std + im_mean

        return image

'''