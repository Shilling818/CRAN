# -*- coding: UTF-8 -*-
"""
Arthur: Shilei Fu
Date: 2018.11.05
Aim: SAR2OPT OPT2SAR

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os, json
import scipy.misc
from os.path import dirname, join
import scipy.io
import random
import time
import shutil
from tensorflow.python import pywrap_tensorflow
from model import Model
import loss

# 只显示 warning 和 Error
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

EPS = 1e-12
ngf = 50
ndf = 64


class create_model(Model):

    def __init__(self, param):
        self.absolute_path = param['absolute_path']
        self.l1_weight = param['l1_weight']
        self.gan_weight = param['gan_weight']
        self.IMG_HEIGHT = param['IMG_HEIGHT']
        self.IMG_WIDTH = param['IMG_WIDTH']
        self.OPT_IMG_CHANNELS = param['OPT_IMG_CHANNELS']
        self.SAR_IMG_CHANNELS = param['SAR_IMG_CHANNELS']
        self._base_lr = param['base_lr']
        self.beta1 = param['beta1']
        self.batch_size = param['batch_size']
        self.sn = param['SN']
        self._max_step = param['Max_Step']
        super(create_model, self).__init__()

    def build_generator(self, im_batch, out_channels, is_training, scope, reuse):
        tmp = []
        with tf.variable_scope(scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            # 256, 256, ngf
            conv1_1 = self.batchnorm(self.conv(im_batch, ngf, 3, 1, sn=self.sn, scope=scope + '_encoder1_1'), is_training, scope + '_encoder1_1')
            conv2_1 = self.batchnorm(self.conv(self.lrelu(conv1_1), ngf, 3, 1, sn=self.sn, scope=scope + '_encoder2_1'), is_training, scope + '_encoder2_1')
            # tmp.append(conv2_1)

            # 128, 128, ngf * 2
            conv1_2 = self.batchnorm(self.conv(self.lrelu(conv2_1), ngf * 2, 3, 2, sn=self.sn, scope=scope + '_encoder1_2'), is_training, scope + '_encoder1_2')
            conv2_2 = self.batchnorm(self.conv(self.lrelu(conv1_2), ngf * 2, 3, 1, sn=self.sn, scope=scope + '_encoder2_2'), is_training, scope + '_encoder2_2')
            # tmp.append(conv2_2)

            # 64, 64, ngf * 4
            conv1_3 = self.batchnorm(self.conv(self.lrelu(conv2_2), ngf * 4, 3, 2, sn=self.sn, scope=scope + '_encoder1_3'), is_training, scope + '_encoder1_3')
            conv2_3 = self.batchnorm(self.conv(self.lrelu(conv1_3), ngf * 4, 3, 1, sn=self.sn, scope=scope + '_encoder2_3'), is_training, scope + '_encoder2_3')
            # tmp.append(conv2_3)

            # 32, 32, ngf * 8
            conv1_4 = self.batchnorm(self.conv(self.lrelu(conv2_3), ngf * 8, 3, 2, sn=self.sn, scope=scope + '_encoder1_4'), is_training, scope + '_encoder1_4')
            conv2_4 = self.batchnorm(self.conv(self.lrelu(conv1_4), ngf * 8, 3, 1, sn=self.sn, scope=scope + '_encoder2_4'), is_training, scope + '_encoder2_4')
            # tmp.append(conv2_4)

            # 16, 16, ngf * 16
            conv1_5 = self.batchnorm(self.conv(self.lrelu(conv2_4), ngf * 16, 3, 2, sn=self.sn, scope=scope + '_encoder1_5'), is_training, scope + '_encoder1_5')
            conv2_5 = self.batchnorm(self.conv(self.lrelu(conv1_5), ngf * 16, 3, 1, sn=self.sn, scope=scope + '_encoder2_5'), is_training, scope + '_encoder2_5')
            tmp.append(conv2_5)

            # 8, 8, ngf * 16
            conv1_6 = self.batchnorm(self.conv(self.lrelu(conv2_5), ngf * 16, 3, 2, sn=self.sn, scope=scope + '_encoder1_6'), is_training, scope + '_encoder1_6')
            conv2_6 = self.batchnorm(self.conv(self.lrelu(conv1_6), ngf * 16, 3, 1, sn=self.sn, scope=scope + '_encoder2_6'), is_training, scope + '_encoder2_6')
            tmp.append(conv2_6)

            bottom = conv2_6
            deconv1_6 = self.batchnorm(self.deconv(self.lrelu(bottom), ngf * 16, 3, 1, sn=self.sn, scope=scope + '_decoder1_6'), is_training, scope + '_decoder1_6')
            resize6 = tf.image.resize_bilinear(im_batch, (self.IMG_HEIGHT // 32, self.IMG_WIDTH // 32), align_corners=True)
            bottom = tf.concat((resize6, deconv1_6), axis=3)
            # 16, 16, ngf * 16
            deconv2_6 = self.batchnorm(self.deconv(self.lrelu(bottom), ngf * 16, 3, 2, sn=self.sn, scope=scope + '_decoder2_6'), is_training, scope + '_decoder2_6')

            bottom = tf.concat((conv1_5, deconv2_6), axis=3)
            deconv1_5 = self.batchnorm(self.deconv(self.lrelu(bottom), ngf * 16, 3, 1, sn=self.sn, scope=scope + '_decoder1_5'), is_training, scope + '_decoder1_5')
            resize5 = tf.image.resize_bilinear(im_batch, (self.IMG_HEIGHT // 16, self.IMG_WIDTH // 16), align_corners=True)
            bottom = tf.concat((resize5, deconv1_5), axis=3)
            # 32, 32, ngf * 8
            deconv2_5 = self.batchnorm(self.deconv(self.lrelu(bottom), ngf * 8, 3, 2, sn=self.sn, scope=scope + '_decoder2_5'), is_training, scope + '_decoder2_5')

            bottom = tf.concat((conv1_4, deconv2_5), axis=3)
            deconv1_4 = self.batchnorm(self.deconv(self.lrelu(bottom), ngf * 8, 3, 1, sn=self.sn, scope=scope + '_decoder1_4'), is_training, scope + '_decoder1_4')
            resize4 = tf.image.resize_bilinear(im_batch, (self.IMG_HEIGHT // 8, self.IMG_WIDTH // 8), align_corners=True)
            bottom = tf.concat((resize4, deconv1_4), axis=3)
            # 64, 64, ngf * 4
            deconv2_4 = self.batchnorm(self.deconv(self.lrelu(bottom), ngf * 4, 3, 2, sn=self.sn, scope=scope + '_decoder2_4'), is_training, scope + '_decoder2_4')

            bottom = tf.concat((conv1_3, deconv2_4), axis=3)
            deconv1_3 = self.batchnorm(self.deconv(self.lrelu(bottom), ngf * 4, 3, 1, sn=self.sn, scope=scope + '_decoder1_3'), is_training, scope + '_decoder1_3')
            resize3 = tf.image.resize_bilinear(im_batch, (self.IMG_HEIGHT // 4, self.IMG_WIDTH // 4), align_corners=True)
            bottom = tf.concat((resize3, deconv1_3), axis=3)
            # 128, 128, ngf * 2
            deconv2_3 = self.batchnorm(self.deconv(self.lrelu(bottom), ngf * 2, 3, 2, sn=self.sn, scope=scope + '_decoder2_3'), is_training, scope + '_decoder2_3')

            bottom = tf.concat((conv1_2, deconv2_3), axis=3)
            deconv1_2 = self.batchnorm(self.deconv(self.lrelu(bottom), ngf * 2, 3, 1, sn=self.sn, scope=scope + '_decoder1_2'), is_training, scope + '_decoder1_2')
            resize2 = tf.image.resize_bilinear(im_batch, (self.IMG_HEIGHT // 2, self.IMG_WIDTH // 2), align_corners=True)
            bottom = tf.concat((resize2, deconv1_2), axis=3)
            # 256, 256, ngf
            deconv2_2 = self.batchnorm(self.deconv(self.lrelu(bottom), ngf, 3, 2, sn=self.sn, scope=scope + '_decoder2_2'), is_training, scope + '_decoder2_2')

            bottom = tf.concat((conv1_1, deconv2_2), axis=3)
            deconv1_1 = self.batchnorm(self.deconv(self.lrelu(bottom), ngf, 3, 1, sn=self.sn, scope=scope + '_decoder1_1'), is_training, scope + '_decoder1_1')
            resize = tf.image.resize_bilinear(im_batch, (self.IMG_HEIGHT, self.IMG_WIDTH), align_corners=True)
            bottom = tf.concat((resize, deconv1_1), axis=3)
            # 256, 256, out_channels
            deconv2_1 = self.deconv(self.lrelu(bottom), out_channels, 3, 1, sn=self.sn, scope=scope + '_decoder2_1')
            # pred = (tf.nn.tanh(deconv2_1) + 1.0) * 127.5
            pred = tf.nn.tanh(deconv2_1)
        return pred, tmp

    def build_discriminator(self, im_batch, is_training, scope, reuse):
        with tf.variable_scope(scope):
            if reuse is True:
                tf.get_variable_scope().reuse_variables()
            # layer_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ndf]
            convolved = self.dis_conv(im_batch, ndf, kernel=4, stride=2, use_bias=True, sn=self.sn, scope=scope + '_discrim1')
            rectified = self.lrelu(convolved)

            # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
            convolved = self.dis_conv(rectified, ndf * 2, kernel=4, stride=2, use_bias=True, sn=self.sn, scope=scope + '_discrim2')
            rectified = self.lrelu(self.batchnorm(convolved, is_training, scope=scope + '_discrim1_2'))
            # rectified = self.lrelu(convolved)

            # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
            convolved = self.dis_conv(rectified, ndf * 4, kernel=4, stride=2, use_bias=True, sn=self.sn, scope=scope + '_discrim3')
            rectified = self.lrelu(self.batchnorm(convolved, is_training, scope=scope + '_discrim1_3'))
            # rectified = self.lrelu(convolved)

            # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
            convolved = self.dis_conv(rectified, ndf * 8, kernel=4, stride=1, use_bias=True, sn=self.sn, scope=scope + '_discrim4')
            rectified = self.lrelu(self.batchnorm(convolved, is_training, scope=scope + '_discrim1_4'))
            # rectified = self.lrelu(convolved)

            # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
            convolved = self.dis_conv(rectified, 1, kernel=4, stride=1, use_bias=True, sn=self.sn, scope=scope + '_discrim5')
            rectified = tf.sigmoid(convolved)
        return rectified

    def model_setup(self):
        self.input_a = tf.placeholder(
            tf.float32, [
                None,
                self.IMG_HEIGHT,
                self.IMG_WIDTH,
                self.OPT_IMG_CHANNELS
            ], name="input_A")

        self.input_b = tf.placeholder(
            tf.float32, [
                None,
                self.IMG_HEIGHT,
                self.IMG_WIDTH,
                self.SAR_IMG_CHANNELS
            ], name="input_B")

        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="lr")
        self.learning_rate2 = tf.placeholder(tf.float32, shape=[], name="lr2")
        self.is_training = tf.placeholder(tf.bool, name='is_training')

    def list_add(self, im_batch):
        assert(len(im_batch) == 4)
        c = [(im_batch[0][i] + im_batch[1][i] + im_batch[2][i] + im_batch[3][i]) / 4.0 for i in range(0, len(im_batch[0]))]
        return c

    def compute_loss(self, im_batch):
        # im_batch[]: input_a, input_b, fake_a, fake_b, predict_fake_a, predict_fake_b,
        #             predict_real_a, predict_real_b, tmp_ab, tmp_ba
        losses = []
        with tf.name_scope("generator_loss"):
            # fake
            gen_loss_fake = (loss.L1_loss(im_batch[0], im_batch[2]) + loss.L1_loss(im_batch[1], im_batch[3])) / 2.
            # fake gan
            # gen_loss_gan = (loss.lsgan_loss_generator(im_batch[4]) + loss.lsgan_loss_generator(im_batch[5])) / 2.
            gen_loss_gan = (loss.gan_loss_generator(im_batch[4]) + loss.gan_loss_generator(im_batch[5])) / 2.
            # latent loss
            gen_loss_latent = tf.constant(0, dtype=tf.float32)
            for i in range(0, len(im_batch[8])):
                gen_loss_latent += loss.L1_loss(im_batch[8][i], im_batch[9][i])
            gen_loss_latent = gen_loss_latent / len(im_batch[8])
            # generator loss
            gen_loss = gen_loss_gan * self.gan_weight + gen_loss_fake * self.l1_weight  # + gen_loss_latent * 25.

            losses = losses + [gen_loss_fake, gen_loss_gan, gen_loss_latent, gen_loss]

        with tf.name_scope("discriminator_loss"):
            # discrim_loss_a = loss.lsgan_loss_discriminator(im_batch[6], im_batch[4])
            # discrim_loss_b = loss.lsgan_loss_discriminator(im_batch[7], im_batch[5])
            discrim_loss_a = loss.gan_loss_discriminator(im_batch[6], im_batch[4])
            discrim_loss_b = loss.gan_loss_discriminator(im_batch[7], im_batch[5])
            losses = losses + [discrim_loss_a, discrim_loss_b]

        with tf.name_scope("generator_train"):
            gen_tvars = [var for var in tf.trainable_variables() if 'UNet1' in var.name or 'UNet2' in var.name]
            self.gen_optim = tf.train.AdamOptimizer(self.learning_rate, self.beta1)
            gen_grads_and_vars = self.gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)

        with tf.name_scope("discriminator_train"):
            discrim_tvars_a = [var for var in tf.trainable_variables() if 'Discrim1' in var.name]
            discrim_tvars_b = [var for var in tf.trainable_variables() if 'Discrim2' in var.name]

            self.discrim_optim_a = tf.train.AdamOptimizer(self.learning_rate2, self.beta1)
            discrim_grads_and_vars_a = self.discrim_optim_a.compute_gradients(discrim_loss_a, var_list=discrim_tvars_a)

            self.discrim_optim_b = tf.train.AdamOptimizer(self.learning_rate2, self.beta1)
            discrim_grads_and_vars_b = self.discrim_optim_b.compute_gradients(discrim_loss_b, var_list=discrim_tvars_b)
        return gen_grads_and_vars, discrim_grads_and_vars_a, discrim_grads_and_vars_b, losses

    def train(self):
        self.model_setup()

        with tf.device('/cpu:0'):
            tower_grads = []
            tower_grads_discrim_a = []
            tower_grads_discrim_b = []
            tower_loss = []

            for i in range(4):
                with tf.device('/gpu:%d' % (i)):
                    _a = self.input_a[i:(i+1)]
                    _b = self.input_b[i:(i+1)]

                    # tf.get_variable_scope()
                    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
                        self.fake_ab, self.tmp_ab = self.build_generator(_a, self.SAR_IMG_CHANNELS, self.is_training, 'UNet1', False)
                        self.fake_ba, self.tmp_ba = self.build_generator(_b, self.OPT_IMG_CHANNELS, self.is_training, 'UNet2', False)

                    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
                        self.predict_real_a = self.build_discriminator(_a, self.is_training, scope='Discrim1', reuse=False)
                        self.predict_real_b = self.build_discriminator(_b, self.is_training, scope='Discrim2', reuse=False)

                        self.predict_fake_a = self.build_discriminator(self.fake_ba, self.is_training, scope='Discrim1', reuse=True)
                        self.predict_fake_b = self.build_discriminator(self.fake_ab, self.is_training, scope='Discrim2', reuse=True)

                        im_batch = [_a, _b, self.fake_ba, self.fake_ab, self.predict_fake_a, self.predict_fake_b,
                                    self.predict_real_a, self.predict_real_b, self.tmp_ab, self.tmp_ba]
                    grads, grads_discrim_a, grads_discrim_b, losses = self.compute_loss(im_batch)
                    tower_grads.append(grads)
                    tower_grads_discrim_a.append(grads_discrim_a)
                    tower_grads_discrim_b.append(grads_discrim_b)
                    tower_loss.append(losses)

            with tf.device('/gpu: 0'):
                tower_grads = self.average_gradients(tower_grads)
                self.gen_train = self.gen_optim.apply_gradients(tower_grads)

                tower_grads_discrim_a = self.average_gradients(tower_grads_discrim_a)
                self.discrim_train_a = self.discrim_optim_a.apply_gradients(tower_grads_discrim_a)
                tower_grads_discrim_b = self.average_gradients(tower_grads_discrim_b)
                self.discrim_train_b = self.discrim_optim_b.apply_gradients(tower_grads_discrim_b)

                self.gen_loss_fake, self.gen_loss_gan, self.gen_loss_latent, self.gen_loss, \
                    self.discrim_loss_a, self.discrim_loss_b = self.list_add(tower_loss)[:]

            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            saver = tf.train.Saver()

            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                sess.run(init)

                # path = self.absolute_path + "/ckpt0"
                # ckpt = tf.train.get_checkpoint_state(path)
                # if ckpt:
                #     saver.restore(sess, ckpt.model_checkpoint_path)
                #     print('loaded ' + ckpt.model_checkpoint_path)

                _num_test = 2570
                _num_train = 10284
                count = 1
                self._save_every_iterations = _num_train

                g_loss_fake = np.zeros(self._save_every_iterations // 4, dtype=float)
                g_loss_gan = np.zeros(self._save_every_iterations // 4, dtype=float)
                g_loss_latent = np.zeros(self._save_every_iterations // 4, dtype=float)
                g_loss = np.zeros(self._save_every_iterations // 4, dtype=float)
                d_loss_a = np.zeros(self._save_every_iterations // 4, dtype=float)
                d_loss_b = np.zeros(self._save_every_iterations // 4, dtype=float)

                g_loss_train = np.zeros(self._max_step, dtype=float)

                if not os.path.isdir(self.absolute_path):
                    os.makedirs(self.absolute_path)

                for epoch in range(0, self._max_step):
                    time_start = time.time()

                    if os.path.isdir(self.absolute_path + "/%04d" % epoch):
                        continue
                    print("In the epoch {}".format(epoch))

                    curr_lr = self._base_lr

                    se = np.random.permutation(self._save_every_iterations) + 1
                    for i in range(0, self._save_every_iterations // 4):
                        print("Processing batch {}/{}".format(i, self._save_every_iterations // 4))
                        image_opt = []
                        image_pol = []
                        for j in range(4):
                            image_opt.append(np.float32(scipy.misc.imread("/emwusr/sfu/data/0.51m/train/opt/%.8d.png" % se[i * 4 + j])) / 127.5 - 1.0)
                            image_pol.append(np.expand_dims(np.float32(scipy.misc.imread("/emwusr/sfu/data/0.51m/train/sar_pure/%.8d.png" % se[i * 4 + j])),
                                axis=2) / 127.5 - 1.0)

                        _, _, _, gen_loss_fake, gen_loss_gan, gen_loss_latent, gen_loss, discrim_loss_a, discrim_loss_b = \
                            sess.run([self.gen_train, self.discrim_train_a, self.discrim_train_b, self.gen_loss_fake,
                                self.gen_loss_gan, self.gen_loss_latent, self.gen_loss, self.discrim_loss_a, self.discrim_loss_b],
                                feed_dict={self.input_a: image_opt[0:4], self.input_b: image_pol[0:4],
                                self.learning_rate: curr_lr, self.learning_rate2: curr_lr, self.is_training: True})

                        g_loss_fake[i] = gen_loss_fake
                        g_loss_gan[i] = gen_loss_gan
                        g_loss_latent[i] = gen_loss_latent
                        g_loss[i] = gen_loss
                        d_loss_a[i] = discrim_loss_a
                        d_loss_b[i] = discrim_loss_b

                        print('gen_loss: {}, discrim_loss_a: {}, discrim_loss_b: {}, gen_loss_gan: {}, gen_loss_fake: {}, gen_loss_latent: {}'
                            .format(np.mean(g_loss[np.where(g_loss)]), np.mean(d_loss_a[np.where(d_loss_a)]),
                              np.mean(d_loss_b[np.where(d_loss_b)]), np.mean(g_loss_gan[np.where(g_loss_gan)]),
                              np.mean(g_loss_fake[np.where(g_loss_fake)]), np.mean(g_loss_latent[np.where(g_loss_latent)])))

                    print('time: {}'.format(time.time() - time_start))
                    g_loss_train[epoch] = np.mean(g_loss[np.where(g_loss)])

                    # save necessary files
                    if not os.path.exists(self.absolute_path + "/%.4d" % epoch):
                        os.makedirs(self.absolute_path + "/%.4d" % epoch)

                    target = open(self.absolute_path + "/%.4d/loss.txt" % epoch, 'w')
                    target.write("gen_loss:\t%f\n" % np.mean(g_loss[np.where(g_loss)]))
                    target.write("gen_loss_gan:\t%f\n" % np.mean(g_loss_gan[np.where(g_loss_gan)]))
                    target.write("gen_loss_fake:\t%f\n" % np.mean(g_loss_fake[np.where(g_loss_fake)]))
                    target.write("gen_loss_latent:\t%f\n" % np.mean(g_loss_latent[np.where(g_loss_latent)]))
                    target.write("d_loss_a:\t%f\n" % np.mean(d_loss_a[np.where(d_loss_a)]))
                    target.write("d_loss_b:\t%f\n" % np.mean(d_loss_b[np.where(d_loss_b)]))
                    target.close()

                    if epoch >= 0:
                        if epoch % 2 == 0:
                            path = self.absolute_path + '/ckpt0'
                            if os.path.exists(path):
                                shutil.rmtree(path)
                            os.makedirs(path)
                            saver.save(sess, self.absolute_path + '/ckpt0/model.ckpt')
                        elif epoch % 2 == 1:
                            path = self.absolute_path + '/ckpt1'
                            if os.path.exists(path):
                                shutil.rmtree(path)
                            os.makedirs(path)
                            saver.save(sess, self.absolute_path + '/ckpt1/model.ckpt')

                    if epoch >= 5:
                        path = self.absolute_path + "/%.4d/data.mat" % (epoch - 5)
                        if os.path.exists(path):
                            os.remove(path)

                    # test
                    index = 15
                    data = np.zeros((index * 2, 2, self.IMG_HEIGHT, self.IMG_WIDTH, 3), dtype=np.float32)
                    data_index = np.zeros(2 * index)
                    random.seed(550)
                    train_index = random.sample(np.arange(_num_train) + 1, index)
                    test_index = random.sample(np.arange(_num_test) + 1, index)

                    # test train_samples
                    for i in range(0, index):
                        image_opt = []
                        image_pol = []
                        for j in range(4):
                            image_opt.append(np.float32(scipy.misc.imread("/emwusr/sfu/data/0.51m/train/opt/%.8d.png" % train_index[i])) / 127.5 - 1.0)
                            image_pol.append(np.expand_dims(np.float32(scipy.misc.imread("/emwusr/sfu/data/0.51m/train/sar_pure/%.8d.png" % train_index[i])),
                                axis=2) / 127.5 - 1.0)
                        data_index[i] = train_index[i]
                        fake_BA_temp, fake_AB_temp = sess.run([
                            self.fake_ba, self.fake_ab], feed_dict={
                            self.input_a: image_opt[0:4], self.input_b: image_pol[0:4], self.is_training: True})

                        # save the images
                        data[i, 0, :, :, :] = (np.squeeze(fake_BA_temp[0]) + 1) * 127.5
                        data[i, 1, :, :, :] = np.repeat(np.expand_dims((np.squeeze(fake_AB_temp[0]) + 1) * 127.5, axis=2), 3, axis=2)

                    # test test_samples
                    for i in range(0, index):
                        image_opt = []
                        image_pol = []
                        for j in range(4):
                            image_opt.append(np.float32(scipy.misc.imread("/emwusr/sfu/data/0.51m/test/opt/%.8d.png" % test_index[i])) / 127.5 - 1.0)
                            image_pol.append(np.expand_dims(np.float32(scipy.misc.imread("/emwusr/sfu/data/0.51m/test/sar_pure/%.8d.png" % test_index[i])),
                                axis=2) / 127.5 - 1.0)

                        data_index[i + index] = test_index[i]

                        fake_BA_temp, fake_AB_temp = sess.run([
                            self.fake_ba, self.fake_ab], feed_dict={
                            self.input_a: image_opt[0:4], self.input_b: image_pol[0:4], self.is_training: True})

                        # save the images
                        data[i + index, 0, :, :, :] = (np.squeeze(fake_BA_temp[0]) + 1) * 127.5
                        data[i + index, 1, :, :, :] = np.repeat(np.expand_dims((np.squeeze(fake_AB_temp[0]) + 1) * 127.5, axis=2), 3, axis=2)
                    scipy.io.savemat(self.absolute_path + '/%.4d/data.mat' % epoch,
                                     {'data': data, 'data_index': data_index})

                    # write the loss of test samples
                    gen_test_loss = np.zeros(_num_test // 4, dtype=float)
                    gen_test_loss_gan = np.zeros(_num_test // 4, dtype=float)
                    gen_test_loss_fake = np.zeros(_num_test // 4, dtype=float)
                    gen_test_loss_latent = np.zeros(_num_test // 4, dtype=float)
                    for i in range(_num_test // 4):
                        image_opt = []
                        image_pol = []
                        for j in range(4):
                            image_opt.append(np.float32(scipy.misc.imread("/emwusr/sfu/data/0.51m/test/opt/%.8d.png" % (i * 4 + j + 1))) / 127.5 - 1.0)
                            image_pol.append(np.expand_dims(np.float32(scipy.misc.imread("/emwusr/sfu/data/0.51m/test/sar_pure/%.8d.png" % (i * 4 + j + 1))),
                                axis=2) / 127.5 - 1.0)

                        gen_loss, gen_loss_gan, gen_loss_fake, gen_loss_latent = sess.run(
                            [self.gen_loss, self.gen_loss_gan, self.gen_loss_fake, self.gen_loss_latent], feed_dict={
                                self.input_a: image_opt[0:4], self.input_b: image_pol[0:4], self.is_training: True})
                        gen_test_loss[i] = gen_loss
                        gen_test_loss_gan[i] = gen_loss_gan
                        gen_test_loss_fake[i] = gen_loss_fake
                        gen_test_loss_latent[i] = gen_loss_latent

                    target = open(self.absolute_path + "/%.4d/test_loss.txt" % epoch, 'w')
                    target.write("gen_loss:\t%f\n" % np.mean(gen_test_loss[np.where(gen_test_loss)]))
                    target.write("gen_loss_gan:\t%f\n" % np.mean(gen_test_loss_gan[np.where(gen_test_loss_gan)]))
                    target.write("gen_loss_fake:\t%f\n" % np.mean(gen_test_loss_fake[np.where(gen_test_loss_fake)]))
                    target.write("gen_loss_latent:\t%f\n" % np.mean(gen_test_loss_latent[np.where(gen_test_loss_latent)]))
                    target.close()

                    # early stop
                    if epoch >= 3:
                        tmp = np.amin(g_loss_train[np.where(g_loss_train)])
                        if tmp < g_loss_train[epoch]:
                            count = count + 1
                        else:
                            count = 1
                        if count >= 5:
                            break


def main():
    # with tf.name_scope('SAR2OPT_OPT2SAR'):
    param = dict()
    param['l1_weight'] = 100.0
    param['gan_weight'] = 5.0
    param['beta1'] = 0.5
    param['batch_size'] = 1
    param['IMG_WIDTH'] = 256
    param['IMG_HEIGHT'] = 256
    param['OPT_IMG_CHANNELS'] = 3
    param['SAR_IMG_CHANNELS'] = 1
    param['base_lr'] = 0.0002  # 0.0001
    param['Max_Step'] = 185
    param['SN'] = False
    param['absolute_path'] = "/emwusr/sfu/result/pix2pix"
    model = create_model(param)
    model.train()


if __name__ == '__main__':
    main()