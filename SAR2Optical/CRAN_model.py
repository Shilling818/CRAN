# -*- coding: UTF-8 -*-
import tensorflow as tf


class Model(object):

    def __init__(self):
        self.weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)

    def lrelu(self, bottom, a=0.2):
        with tf.name_scope("lrelu"):
            bottom = tf.identity(bottom)
            return (0.5 * (1 + a)) * bottom + (0.5 * (1 - a)) * tf.abs(bottom)

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv(self, bottom, channels, kernel=3, stride=2, use_bias=True, sn=False, scope='conv_0'):
        with tf.variable_scope(scope):
            if sn:
                w = tf.get_variable("kernel", shape=[kernel, kernel, bottom.get_shape()[-1], channels],
                                    initializer=self.weight_init,
                                    regularizer=tf.contrib.layers.l2_regularizer(0.003))
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                w_sn = self.spectral_norm(w)
                bottom = tf.nn.conv2d(input=bottom, filter=w_sn,
                                 strides=[1, stride, stride, 1], padding='SAME')
                if use_bias:
                    bottom = tf.nn.bias_add(bottom, bias)
            else:
                bottom = tf.layers.conv2d(inputs=bottom, filters=channels,
                                     kernel_size=kernel, kernel_initializer=self.weight_init,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                                     strides=(stride, stride), padding='SAME', use_bias=use_bias)
            return bottom

    def deconv(self, bottom, channels, kernel=3, stride=2, use_bias=True, sn=False, scope='deconv_0'):
        with tf.variable_scope(scope):
            bottom_shape = bottom.shape.as_list()
            output_shape = [1, bottom_shape[1] * stride, bottom_shape[2] * stride, channels]
            output_shape = tf.convert_to_tensor(output_shape[:])
            if sn:
                w = tf.get_variable("kernel", shape=[kernel, kernel, channels, bottom.get_shape()[-1]],
                                    initializer=self.weight_init, regularizer=tf.contrib.layers.l2_regularizer(0.003))
                w_sn = self.spectral_norm(w)
                bottom = tf.nn.conv2d_transpose(value=bottom, filter=w_sn, output_shape=output_shape,
                                           strides=[1, stride, stride, 1], padding='SAME')

                if use_bias:
                    bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                    bottom = tf.nn.bias_add(bottom, bias)
            else:
                bottom = tf.layers.conv2d_transpose(inputs=bottom, filters=channels,
                                               kernel_size=kernel, kernel_initializer=self.weight_init,
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                                               strides=(stride, stride), padding='SAME', use_bias=use_bias)
            return bottom

    def dis_conv(self, bottom, channels, kernel=4, stride=2, use_bias=True, sn=False, scope='conv_0'):
        with tf.variable_scope(scope):
            if sn:
                w = tf.get_variable("kernel", shape=[kernel, kernel, bottom.get_shape()[-1], channels],
                                    initializer=self.weight_init,
                                    regularizer=tf.contrib.layers.l2_regularizer(0.003))
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                w_sn = self.spectral_norm(w)
                # tf.add_to_collection(scope.split('_')[0] + "_loss", self.ortho_norm(w_sn))
                bottom = tf.nn.conv2d(input=bottom, filter=w_sn, strides=[1, stride, stride, 1], padding='SAME')
                if use_bias:
                    bottom = tf.nn.bias_add(bottom, bias)
            else:
                bottom = tf.layers.conv2d(inputs=bottom, filters=channels,
                                     kernel_size=kernel, kernel_initializer=self.weight_init,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                                     strides=(stride, stride), padding='SAME', use_bias=use_bias)
            return bottom

    def batchnorm(self, bottom, is_training, scope):
        with tf.variable_scope(scope):
            return tf.layers.batch_normalization(bottom, axis=3, epsilon=1e-5, momentum=0.1, training=is_training,
                                                 gamma_initializer=tf.random_normal_initializer(1.0, 0.02))

    def spectral_norm(self, w, iteration=1):
        # tensor 可以使用as_list？
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])
        # Gets an existing variable with these parameters or create a new one.
        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

        u_hat = u
        v_hat = None
        for i in range(iteration):
            """
            power iteration
            Usually iteration = 1 will be enough
            """
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = self.l2_norm(v_)

            u_ = tf.matmul(v_hat, w)
            u_hat = self.l2_norm(u_)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
        w_norm = w / sigma

        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = tf.reshape(w_norm, w_shape)
        return w_norm

    def ortho_norm(self, w):
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])
        I = tf.eye(w_shape[-1])
        x = tf.multiply(tf.matmul(tf.transpose(w), w), 1 - I)
        return tf.reduce_mean(x)

    def l2_norm(self, v, eps=1e-12):
        return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

    def average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads
