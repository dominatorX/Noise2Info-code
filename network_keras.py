import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from network_configure import conf_basic_ops
import numpy as np
import tensorflow as tf


"""This script generates the U-Net architecture according to conf_unet.
"""


class CustomModel(tf.keras.Model):
    def __init__(self, net, dimension, in_channels, masking, num_mc, removed_noise, writer=None):
        super(CustomModel, self).__init__()
        self.dimension = dimension
        self.in_channels = in_channels
        self.sgml = tf.Variable(1., trainable=False)
        self.masking = masking
        self.net = net
        self.num_mc = num_mc
        self.removed_noise = removed_noise
        self.writer = writer

    def image_summary(self, img):
        return tf.reduce_max(img, axis=1) if self.dim == 3 else img

    # Local average excluding the center pixel (donut)
    def mask_kernel(self, features):
        kernel = (np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]])
                  if self.dimension == 2 else
                  np.array([[[0, 0.5, 0], [0.5, 1.0, 0.5], [0, 0.5, 0]],
                            [[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]],
                            [[0, 0.5, 0], [0.5, 1.0, 0.5], [0, 0.5, 0]]]))
        kernel = (kernel / kernel.sum())
        if self.dimension == 2:
            kernels = np.empty([3, 3, self.in_channels, self.in_channels])
            for i in range(self.in_channels):
                kernels[:, :, i, i] = kernel
        else:
            kernels = np.empty([3, 3, 3, self.in_channels, self.in_channels])
            for i in range(self.in_channels):
                kernels[:, :, :, i, i] = kernel
        nn_conv_op = tf.nn.conv2d if self.dimension == 2 else tf.nn.conv3d
        return nn_conv_op(features, tf.constant(kernels.astype('float32')),
                          [1] * self.dimension + [1, 1], padding='SAME')

    def train_step(self, data):
        features, labels = data

        with tf.GradientTape() as tape:
            noise, mask = tf.split(labels, [self.in_channels, self.in_channels], -1)

            if self.masking == 'gaussian':
                masked_features = (1 - mask) * features + mask * noise
            elif self.masking == 'donut':
                masked_features = (1 - mask) * features + mask * self.mask_kernel(features)
            else:
                raise NotImplementedError

            preds = self(masked_features, training=True)
            rawpreds = self(features, training=True)

            # Loss components
            n_mask = tf.reduce_sum(mask)
            rec_mse = tf.reduce_mean(tf.square(rawpreds - features), axis=None)
            inv_mse = tf.reduce_sum(tf.square(rawpreds - preds) * mask) / n_mask
            bsp_mse = tf.reduce_sum(tf.square(features - preds) * mask) / n_mask
            loss = rec_mse + 2 * self.sgml * tf.sqrt(inv_mse)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(rawpreds, features)
        return {m.name: m.result() for m in self.metrics}

    def update_feed(self, features, labels, batch_size):
        noise, mask = tf.split(labels, [self.in_channels, self.in_channels], -1)

        if self.masking == 'gaussian':
            masked_features = (1 - mask) * features + mask * noise
        elif self.masking == 'donut':
            masked_features = (1 - mask) * features + mask * self.mask_kernel(features)
        else:
            raise NotImplementedError

        preds = self(masked_features, training=False)
        rawpreds = self(features, training=False)

        # Loss components
        n_mask = tf.reduce_sum(mask)
        rec_mse = tf.reduce_mean(tf.square(rawpreds - features), axis=None)
        inv_mse = tf.reduce_sum(tf.square(rawpreds - preds) * mask) / n_mask
        bsp_mse = tf.reduce_sum(tf.square(features - preds) * mask) / n_mask
        return rec_mse.numpy(), inv_mse.numpy() * n_mask.numpy(), n_mask.numpy(), \
               np.sort((rawpreds - features).numpy().reshape((batch_size, -1)))

    def update_sgml(self, E_fx_x, E_f_xj_fx_j, sgm_r=0):
        sgml_ = (E_f_xj_fx_j)**0.5 + (E_f_xj_fx_j + E_fx_x-sgm_r)**0.5
        print('new sigma_loss is ', sgml_)
        if 0 < sgml_ < self.sgml.numpy():  # 0 < in case of nan
            self.sgml.assign(float(sgml_))
            print('sigma_loss updated to ', self.sgml.numpy())

    def update(self, update_data, update_samples, batch_size, total_dims):
        step, rec, inv, n_mask, sgm_r = 0, 0, 0, 0, 0
        for features, labels in update_data:
            if step == update_samples: break
            step += 1
            rec_, inv_, n_mask_, removed_noise = self.update_feed(features, labels, batch_size)
            rec += rec_
            inv += inv_
            n_mask += n_mask_
            self.removed_noise[(step - 1) * batch_size: step * batch_size] = removed_noise
        for _ in range(self.num_mc):
            samples = np.sort(np.random.choice(self.removed_noise.reshape(update_samples * batch_size * total_dims),
                                               [1, total_dims]))
            sgm_r += np.square(samples - self.removed_noise).mean((0, 1))
        self.update_sgml(rec / update_samples, inv / n_mask, sgm_r=sgm_r / self.num_mc)

    def call(self, x):
        return self.net(x)


class res_block(tf.keras.layers.Layer):
    def __init__(self, output_filters, dimension):
        super(res_block, self).__init__()
        self.output_filters = output_filters
        conv = tf.keras.layers.Conv2D if dimension == '2D' else tf.keras.layers.Conv3D
        self.projection_shortcut = conv(output_filters, 1, 1, padding='same', use_bias=False,
                                        kernel_initializer=conf_basic_ops['kernel_initializer'])
        self.convolution_1 = conv(output_filters, 3, 1, padding='same', use_bias=False,
                                        kernel_initializer=conf_basic_ops['kernel_initializer'])
        self.convolution_2 = conv(output_filters, 3, 1, padding='same', use_bias=False,
                                        kernel_initializer=conf_basic_ops['kernel_initializer'])
        self.relu_1 = tf.keras.layers.Activation('relu')
        self.relu_2 = tf.keras.layers.Activation('relu')
        self.norm_1 = tf.keras.layers.BatchNormalization()
        self.norm_2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        if inputs.shape[-1] == self.output_filters:
            shortcut = inputs
            inputs = self.norm_1(inputs)
            inputs = self.relu_1(inputs)
        else:
            inputs = self.norm_1(inputs)
            inputs = self.relu_1(inputs)
            shortcut = self.projection_shortcut(inputs)
        inputs = self.convolution_1(inputs)
        inputs = self.norm_2(inputs)
        inputs = self.relu_2(inputs)
        inputs = self.convolution_2(inputs)
        return tf.add(shortcut, inputs)


class down_res_block(tf.keras.layers.Layer):
    def __init__(self, output_filters, dimension):
        super(down_res_block, self).__init__()
        self.output_filters = output_filters
        conv = tf.keras.layers.Conv2D if dimension == '2D' else tf.keras.layers.Conv3D
        self.projection_shortcut = conv(output_filters, 1, 2, padding='same', use_bias=False,
                                        kernel_initializer=conf_basic_ops['kernel_initializer'])
        self.convolution_1 = conv(output_filters, 2, 2, padding='same', use_bias=False,
                                  kernel_initializer=conf_basic_ops['kernel_initializer'])
        self.convolution_2 = conv(output_filters, 2, 2, padding='same', use_bias=False,
                                  kernel_initializer=conf_basic_ops['kernel_initializer'])
        self.relu_1 = tf.keras.layers.Activation('relu')
        self.relu_2 = tf.keras.layers.Activation('relu')
        self.norm_1 = tf.keras.layers.BatchNormalization()
        self.norm_2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        inputs = self.norm_1(inputs)
        inputs = self.relu_1(inputs)
        shortcut = self.projection_shortcut(inputs)
        inputs = self.convolution_1(inputs)
        inputs = self.norm_2(inputs)
        inputs = self.relu_2(inputs)
        inputs = self.convolution_2(inputs)
        return tf.add(shortcut, inputs)


class UNet(tf.keras.layers.Layer):
    def __init__(self, conf_unet):
        super(UNet, self).__init__()
        self.depth = conf_unet['depth']
        self.dimension = conf_unet['dimension']
        self.first_output_filters = conf_unet['first_output_filters']
        self.encoding_block_sizes = conf_unet['encoding_block_sizes']
        self.downsampling = conf_unet['downsampling']
        self.decoding_block_sizes = conf_unet['decoding_block_sizes']
        self.skip_method = conf_unet['skip_method']
        self.convolution = tf.keras.layers.Conv2D if self.dimension == '2D' else tf.keras.layers.Conv3D

        self.first_convolution = self.convolution(self.first_output_filters, 3, 1, padding='same', use_bias=False,
                                                  kernel_initializer=conf_basic_ops['kernel_initializer'])
        for block_index in range(0, self.encoding_block_sizes[0]):
            setattr(self, 'res1_' + str(block_index), res_block(self.first_output_filters, self.dimension))

        for i in range(2, self.depth+1):
            output_filters = self.first_output_filters * (2**(i-1))

            setattr(self, 'encoding_block_downsampling_'+str(i),
                    self._get_downsampling_function(self.downsampling[i-2], output_filters, self.dimension))

            for block_index in range(0, self.encoding_block_sizes[i-1]):
                setattr(self, 'encoding_block_res'+str(i)+'_' + str(block_index),
                        res_block(output_filters, self.dimension))

        output_filters = self.first_output_filters * (2 ** (self.depth - 1))
        for block_index in range(0, 1):
            current_func = res_block
            setattr(self, 'bottom_block_'+str(block_index),
                    current_func(output_filters, self.dimension))

        for j in range(self.depth-1, 0, -1):
            output_filters = self.first_output_filters * (2**(j-1))
            transposed_convolution = tf.keras.layers.Conv2DTranspose if self.dimension == '2D' else \
                tf.keras.layers.Conv3DTranspose
            setattr(self, 'decoding_block_upsampling_' + str(j),
                    transposed_convolution(output_filters, 2, 2, padding='same', use_bias=True,
                                           kernel_initializer=conf_basic_ops['kernel_initializer']))

            for block_index in range(0, self.decoding_block_sizes[self.depth-1-j]):
                setattr(self, 'decoding_block_res_' + str(j) +'_'+str(block_index),
                        res_block(output_filters, self.dimension))

    def call(self, inputs):
        """Add operations to classify a batch of input images.

        Args:
            inputs: A Tensor representing a batch of input images.

        Returns:
            A logits Tensor with shape [<batch_size>, self.num_classes].
        """

        return self._build_network(inputs)


    ################################################################################
    # Composite blocks building the network
    ################################################################################
    def _build_network(self, inputs):
        # first_convolution
        inputs = self.first_convolution(inputs)

        # encoding_block_1
        for block_index in range(0, self.encoding_block_sizes[0]):
            inputs = getattr(self, 'res1_' + str(block_index))(inputs)

        # encoding_block_i (down) = downsampling + zero or more res_block, i = 2, 3, ..., depth
        skip_inputs = [] # for identity skip connections
        for i in range(2, self.depth+1):
            skip_inputs.append(inputs)
            # downsampling
            inputs = getattr(self, 'encoding_block_downsampling_' + str(i))(inputs)

            for block_index in range(0, self.encoding_block_sizes[i - 1]):
                inputs = getattr(self, 'encoding_block_res' + str(i) + '_' + str(block_index))(inputs)

        # bottom_block = a combination of same_gto and res_block
        output_filters = self.first_output_filters * (2**(self.depth-1))
        for block_index in range(0, 1):
            inputs = getattr(self, 'bottom_block_'+str(block_index))(inputs)

        """
        Note: Identity skip connections are between the output of encoding_block_i and
        the output of upsampling in decoding_block_i, i = 1, 2, ..., depth-1.
        skip_inputs[i] is the output of encoding_block_i now.
        len(skip_inputs) == depth - 1
        skip_inputs[depth-2] should be combined during decoding_block_depth-1
        skip_inputs[0] should be combined during decoding_block_1
        """

        # decoding_block_j (up) = upsampling + zero or more res_block, j = depth-1, depth-2, ..., 1
        for j in range(self.depth-1, 0, -1):
            inputs = getattr(self, 'decoding_block_upsampling_' + str(j))(inputs)

            # combine with skip connections
            if self.skip_method == 'add':
                inputs = tf.add(inputs, skip_inputs[j-1])
            elif self.skip_method == 'concat':
                inputs = tf.concat([inputs, skip_inputs[j-1]], axis=-1)

            for block_index in range(0, self.decoding_block_sizes[self.depth-1-j]):
                inputs = getattr(self, 'decoding_block_res_' + str(j) + '_' + str(block_index))(inputs)

        return inputs


    def _get_downsampling_function(self, name, output_filters, dimension):
        if name == 'down_res_block':
            return down_res_block(output_filters, dimension)
        elif name == 'convolution':
            return self.convolution(output_filters, 2, 2, padding='same', use_bias=True,
                                    kernel_initializer=conf_basic_ops['kernel_initializer'])
        else:
            raise ValueError("Unsupported function: %s." % (name))