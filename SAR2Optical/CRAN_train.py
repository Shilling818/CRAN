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
from CRAN_model import Model
from CRAN_loss import *
import argparse
from dataset import *
import h5py
import cv2

# 只显示 warning 和 Error
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

EPS = 1e-12

program_name = 'CRAN Training Program'
desc = 'Program for training CRAN.'
parser = argparse.ArgumentParser(program_name, description=desc)
parser.add_argument('--name', type=str, default='SAR Segmentation', help='Experiment Name')
parser.add_argument('--sample_dir', type=str, default='../dataset/GF3')
parser.add_argument('--save_dir', type=str, default='../results_GF3_new', help='directory to save results')
parser.add_argument('--pretrained_weights_dir', type=str, default=None)
parser.add_argument('--IMG_HEIGHT', type=int, default=256)
parser.add_argument('--IMG_WIDTH', type=int, default=256)
parser.add_argument('--MAP_IMAGE_CHANNELS', type=int, default=5)
parser.add_argument('--SAR_IMAGE_CHANNELS', type=int, default=3)
parser.add_argument('--OPT_IMAGE_CHANNELS', type=int, default=3)
# building, vegetation, water, road, bare_land
parser.add_argument('--color_map', type=list, default=[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 0, 0]])
parser.add_argument('--learning_rate', type=float, default=0.0002)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--max_epoch', type=int, default=150, help='max epochs to train')
parser.add_argument('--num_save', type=int, default=15)
parser.add_argument('--ngf', type=int, default=50)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--l1_weight', default=100.0, type=float)
parser.add_argument('--gan_weight', default=5.0, type=float)
parser.add_argument('--beta1', default=0.5, type=float, help='''Adam Optimizer''')
parser.add_argument('--num_gpus', default=4, type=int)
parser.add_argument('--sn', default=False, type=bool)
params = parser.parse_args()


def _norm_image_inv(image):
    return (image + 1.) * 127.5


class create_model(Model):

    def __init__(self, params):
        self.params = params
        super(create_model, self).__init__()

    def build_generator(self, im_batch, out_channels, is_training, scope, reuse):
        tmp = []
        with tf.variable_scope(scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            # 256, 256, ngf
            conv1_1 = self.conv(im_batch, self.params.ngf, 3, 1, sn=self.params.sn, scope=scope + '_encoder1_1')
            conv2_1 = self.conv(self.lrelu(conv1_1), self.params.ngf, 3, 1, sn=self.params.sn, scope=scope + '_encoder2_1')
            # tmp.append(conv2_1)

            # 128, 128, ngf * 2
            conv1_2 = self.conv(self.lrelu(conv2_1), self.params.ngf * 2, 3, 2, sn=self.params.sn, scope=scope + '_encoder1_2')
            conv2_2 = self.conv(self.lrelu(conv1_2), self.params.ngf * 2, 3, 1, sn=self.params.sn, scope=scope + '_encoder2_2')
            # tmp.append(conv2_2)

            # 64, 64, ngf * 4
            conv1_3 = self.conv(self.lrelu(conv2_2), self.params.ngf * 4, 3, 2, sn=self.params.sn, scope=scope + '_encoder1_3')
            conv2_3 = self.conv(self.lrelu(conv1_3), self.params.ngf * 4, 3, 1, sn=self.params.sn, scope=scope + '_encoder2_3')
            # tmp.append(conv2_3)

            # 32, 32, ngf * 8
            conv1_4 = self.conv(self.lrelu(conv2_3), self.params.ngf * 8, 3, 2, sn=self.params.sn, scope=scope + '_encoder1_4')
            conv2_4 = self.conv(self.lrelu(conv1_4), self.params.ngf * 8, 3, 1, sn=self.params.sn, scope=scope + '_encoder2_4')
            # tmp.append(conv2_4)

            # 16, 16, ngf * 16
            conv1_5 = self.conv(self.lrelu(conv2_4), self.params.ngf * 16, 3, 2, sn=self.params.sn, scope=scope + '_encoder1_5')
            conv2_5 = self.conv(self.lrelu(conv1_5), self.params.ngf * 16, 3, 1, sn=self.params.sn, scope=scope + '_encoder2_5')
            # tmp.append(conv2_5)

            # 8, 8, ngf * 16
            conv1_6 = self.conv(self.lrelu(conv2_5), self.params.ngf * 16, 3, 2, sn=self.params.sn, scope=scope + '_encoder1_6')
            conv2_6 = self.conv(self.lrelu(conv1_6), self.params.ngf * 16, 3, 1, sn=self.params.sn, scope=scope + '_encoder2_6')
            # tmp.append(conv2_6)

            bottom = conv2_6
            deconv1_6 = self.deconv(self.lrelu(bottom), self.params.ngf * 16, 3, 1, sn=self.params.sn, scope=scope + '_decoder1_6')
            tmp_shape = tf.shape(deconv1_6)
            resize6 = tf.image.resize_bilinear(im_batch, (tmp_shape[1], tmp_shape[2]), align_corners=True)
            bottom = tf.concat((resize6, deconv1_6), axis=3)
            # 16, 16, ngf * 16
            deconv2_6 = self.deconv(self.lrelu(bottom), self.params.ngf * 16, 3, 2, sn=self.params.sn, scope=scope + '_decoder2_6')

            bottom = tf.concat((conv1_5, deconv2_6), axis=3)
            deconv1_5 = self.deconv(self.lrelu(bottom), self.params.ngf * 16, 3, 1, sn=self.params.sn, scope=scope + '_decoder1_5')
            tmp_shape = tf.shape(deconv1_5)
            resize5 = tf.image.resize_bilinear(im_batch, (tmp_shape[1], tmp_shape[2]), align_corners=True)
            bottom = tf.concat((resize5, deconv1_5), axis=3)
            # 32, 32, ngf * 8
            deconv2_5 = self.deconv(self.lrelu(bottom), self.params.ngf * 8, 3, 2, sn=self.params.sn, scope=scope + '_decoder2_5')

            bottom = tf.concat((conv1_4, deconv2_5), axis=3)
            deconv1_4 = self.deconv(self.lrelu(bottom), self.params.ngf * 8, 3, 1, sn=self.params.sn, scope=scope + '_decoder1_4')
            tmp_shape = tf.shape(deconv1_4)
            resize4 = tf.image.resize_bilinear(im_batch, (tmp_shape[1], tmp_shape[2]), align_corners=True)
            bottom = tf.concat((resize4, deconv1_4), axis=3)
            # 64, 64, ngf * 4
            deconv2_4 = self.deconv(self.lrelu(bottom), self.params.ngf * 4, 3, 2, sn=self.params.sn, scope=scope + '_decoder2_4')

            bottom = tf.concat((conv1_3, deconv2_4), axis=3)
            deconv1_3 = self.deconv(self.lrelu(bottom), self.params.ngf * 4, 3, 1, sn=self.params.sn, scope=scope + '_decoder1_3')
            tmp_shape = tf.shape(deconv1_3)
            resize3 = tf.image.resize_bilinear(im_batch, (tmp_shape[1], tmp_shape[2]), align_corners=True)
            bottom = tf.concat((resize3, deconv1_3), axis=3)
            # 128, 128, ngf * 2
            deconv2_3 = self.deconv(self.lrelu(bottom), self.params.ngf * 2, 3, 2, sn=self.params.sn, scope=scope + '_decoder2_3')

            bottom = tf.concat((conv1_2, deconv2_3), axis=3)
            deconv1_2 = self.deconv(self.lrelu(bottom), self.params.ngf * 2, 3, 1, sn=self.params.sn, scope=scope + '_decoder1_2')
            tmp_shape = tf.shape(deconv1_2)
            resize2 = tf.image.resize_bilinear(im_batch, (tmp_shape[1], tmp_shape[2]), align_corners=True)
            bottom = tf.concat((resize2, deconv1_2), axis=3)
            # 256, 256, ngf
            deconv2_2 = self.deconv(self.lrelu(bottom), self.params.ngf, 3, 2, sn=self.params.sn, scope=scope + '_decoder2_2')

            bottom = tf.concat((conv1_1, deconv2_2), axis=3)
            deconv1_1 = self.deconv(self.lrelu(bottom), self.params.ngf, 3, 1, sn=self.params.sn, scope=scope + '_decoder1_1')
            tmp_shape = tf.shape(deconv1_1)
            resize = tf.image.resize_bilinear(im_batch, (tmp_shape[1], tmp_shape[2]), align_corners=True)
            bottom = tf.concat((resize, deconv1_1), axis=3)
            # 256, 256, out_channels
            deconv2_1 = self.deconv(self.lrelu(bottom), out_channels, 3, 1, sn=self.params.sn, scope=scope + '_decoder2_1')
            # pred = (tf.nn.tanh(deconv2_1) + 1.0) * 127.5
            pred = tf.nn.tanh(deconv2_1)
        return pred

    def build_discriminator(self, im_batch, is_training, scope, reuse):
        with tf.variable_scope(scope):
            if reuse is True:
                tf.get_variable_scope().reuse_variables()
            # layer_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ndf]
            convolved = self.dis_conv(im_batch, self.params.ndf, kernel=4, stride=2, use_bias=True, sn=self.params.sn, scope=scope + '_discrim1')
            rectified = self.lrelu(convolved)

            # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
            convolved = self.dis_conv(rectified, self.params.ndf * 2, kernel=4, stride=2, use_bias=True, sn=self.params.sn, scope=scope + '_discrim2')
            rectified = self.lrelu(self.batchnorm(convolved, is_training, scope=scope + '_discrim1_2'))
            # rectified = self.lrelu(convolved)

            # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
            convolved = self.dis_conv(rectified, self.params.ndf * 4, kernel=4, stride=2, use_bias=True, sn=self.params.sn, scope=scope + '_discrim3')
            rectified = self.lrelu(self.batchnorm(convolved, is_training, scope=scope + '_discrim1_3'))
            # rectified = self.lrelu(convolved)

            # layer_4: [batch, 32, 32, ndf * 4] => [batch, 32, 32, ndf * 8]
            convolved = self.dis_conv(rectified, self.params.ndf * 8, kernel=4, stride=1, use_bias=True, sn=self.params.sn, scope=scope + '_discrim4')
            rectified = self.lrelu(self.batchnorm(convolved, is_training, scope=scope + '_discrim1_4'))
            # rectified = self.lrelu(convolved)

            # layer_5: [batch, 32, 32, ndf * 8] => [batch, 32, 32, 1]
            convolved = self.dis_conv(rectified, 1, kernel=4, stride=1, use_bias=True, sn=self.params.sn, scope=scope + '_discrim5')
            rectified = tf.sigmoid(convolved)
        return rectified

    def model_setup(self):
        self.input_a = tf.placeholder(
            tf.float32, [
                None,
                self.params.IMG_HEIGHT,
                self.params.IMG_WIDTH,
                self.params.SAR_IMAGE_CHANNELS
            ], name="input_A")

        self.input_b = tf.placeholder(
            tf.float32, [
                None,
                self.params.IMG_HEIGHT,
                self.params.IMG_WIDTH,
                self.params.OPT_IMAGE_CHANNELS
            ], name="input_B")
        self.is_training = tf.placeholder(tf.bool, name='is_training')

    def list_add(self, im_batch):
        assert(len(im_batch) == 4)
        c = [(im_batch[0][i] + im_batch[1][i] + im_batch[2][i] + im_batch[3][i]) / 4.0 for i in range(0, len(im_batch[0]))]
        return c

    def compute_loss(self, im_batch):
        # im_batch[]: input_a, input_b, fake_a, fake_b, predict_fake_a, predict_fake_b,
        #             predict_real_a, predict_real_b
        losses = []
        with tf.name_scope("generator_loss"):
            # fake
            gen_loss_fake = (L1_loss(im_batch[0], im_batch[2]) + L1_loss(im_batch[1], im_batch[3])) / 2.
            # fake gan
            # gen_loss_gan = (loss.lsgan_loss_generator(im_batch[4]) + loss.lsgan_loss_generator(im_batch[5])) / 2.
            gen_loss_gan = (gan_loss_generator(im_batch[4]) + gan_loss_generator(im_batch[5])) / 2.
            # generator loss
            gen_loss = gen_loss_gan * self.params.gan_weight + gen_loss_fake * self.params.l1_weight

            losses = losses + [gen_loss_fake, gen_loss_gan, gen_loss]

        with tf.name_scope("discriminator_loss"):
            discrim_loss_a = gan_loss_discriminator(im_batch[6], im_batch[4])
            discrim_loss_b = gan_loss_discriminator(im_batch[7], im_batch[5])
            losses = losses + [discrim_loss_a, discrim_loss_b]

        with tf.name_scope("generator_train"):
            gen_tvars = [var for var in tf.trainable_variables() if 'CRAN1_G' in var.name or 'CRAN2_G' in var.name]
            self.gen_optim = tf.train.AdamOptimizer(self.params.learning_rate, self.params.beta1)
            gen_grads_and_vars = self.gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)

        with tf.name_scope("discriminator_train"):
            discrim_tvars_a = [var for var in tf.trainable_variables() if 'CRAN2_D' in var.name]
            discrim_tvars_b = [var for var in tf.trainable_variables() if 'CRAN1_D' in var.name]

            self.discrim_optim_a = tf.train.AdamOptimizer(self.params.learning_rate, self.params.beta1)
            discrim_grads_and_vars_a = self.discrim_optim_a.compute_gradients(discrim_loss_a, var_list=discrim_tvars_a)

            self.discrim_optim_b = tf.train.AdamOptimizer(self.params.learning_rate, self.params.beta1)
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
                        self.fake_ab = self.build_generator(_a, self.params.OPT_IMAGE_CHANNELS, self.is_training, 'CRAN1_G', False)
                        self.fake_ba = self.build_generator(_b, self.params.SAR_IMAGE_CHANNELS, self.is_training, 'CRAN2_G', False)

                    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
                        self.predict_real_a = self.build_discriminator(_a, self.is_training, scope='CRAN2_D', reuse=False)
                        self.predict_real_b = self.build_discriminator(_b, self.is_training, scope='CRAN1_D', reuse=False)

                        self.predict_fake_a = self.build_discriminator(self.fake_ba, self.is_training, scope='CRAN2_D', reuse=True)
                        self.predict_fake_b = self.build_discriminator(self.fake_ab, self.is_training, scope='CRAN1_D', reuse=True)

                        im_batch = [_a, _b, self.fake_ba, self.fake_ab, self.predict_fake_a, self.predict_fake_b,
                                    self.predict_real_a, self.predict_real_b]
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

                self.gen_loss_fake, self.gen_loss_gan, self.gen_loss, \
                    self.discrim_loss_a, self.discrim_loss_b = self.list_add(tower_loss)[:]

            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            self.saver = tf.train.Saver()

            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                sess.run(init)

                if self.params.pretrained_weights_dir is not None:
                    ckpt = tf.train.get_checkpoint_state(self.params.pretrained_weights_dir)
                    if ckpt:
                        self.saver.restore(sess, ckpt.model_checkpoint_path)
                        print('loaded ' + ckpt.model_checkpoint_path)

                print('Loading data...')
                train_dataset = dataset_SAR2OPT(os.sep.join([self.params.sample_dir, 'train/sar']),
                                                os.sep.join([self.params.sample_dir, 'train/opt']), self.params, 1000)
                valid_dataset = dataset_SAR2OPT(os.sep.join([self.params.sample_dir, 'test/sar']),
                                                os.sep.join([self.params.sample_dir, 'test/opt']), self.params, 1000)
                print('Loading data done!')
                train_one_element = getone(train_dataset)
                valid_one_element = getone(valid_dataset)

                num_train_dataset = len(os.listdir(os.sep.join([self.params.sample_dir, 'train/sar'])))
                num_valid_dataset = len(os.listdir(os.sep.join([self.params.sample_dir, 'test/sar'])))

                _train_iteration = num_train_dataset // self.params.batch_size
                _valid_iteration = num_valid_dataset // self.params.batch_size

                # save some indexs
                g_loss_fake_train = np.zeros(_train_iteration, dtype=float)
                g_loss_gan_train = np.zeros(_train_iteration, dtype=float)
                g_loss_train = np.zeros(_train_iteration, dtype=float)
                d_loss_train_a = np.zeros(_train_iteration, dtype=float)
                d_loss_train_b = np.zeros(_train_iteration, dtype=float)

                g_loss_valid = np.zeros(_valid_iteration, dtype=float)

                loss_total = np.zeros((self.params.max_epoch, 6), dtype=float)

                count = 0

                if os.path.isdir(self.params.save_dir):
                    shutil.rmtree(self.params.save_dir)
                os.makedirs(self.params.save_dir)

                for epoch in range(0, self.params.max_epoch):
                    time_start = time.time()
                    print("In the epoch {}".format(epoch + 1))

                    for iter in range(0, _train_iteration):
                        value = sess.run(train_one_element)
                        _, _, _, gen_loss_fake, gen_loss_gan, gen_loss, discrim_loss_a, discrim_loss_b = \
                            sess.run([self.gen_train, self.discrim_train_a, self.discrim_train_b, self.gen_loss_fake,
                                      self.gen_loss_gan, self.gen_loss, self.discrim_loss_a, self.discrim_loss_b],
                                     feed_dict={self.input_a: value[0], self.input_b: value[1], self.is_training: True})

                        g_loss_fake_train[iter] = gen_loss_fake
                        g_loss_gan_train[iter] = gen_loss_gan
                        g_loss_train[iter] = gen_loss
                        d_loss_train_a[iter] = discrim_loss_a
                        d_loss_train_b[iter] = discrim_loss_b

                        loss_total[epoch, :5] = [np.mean(d_loss_train_a[np.where(d_loss_train_a)]),
                                                 np.mean(d_loss_train_b[np.where(d_loss_train_b)]),
                                                 np.mean(g_loss_train[np.where(g_loss_train)]),
                                                 np.mean(g_loss_fake_train[np.where(g_loss_fake_train)]),
                                                 np.mean(g_loss_gan_train[np.where(g_loss_gan_train)])]

                        if iter % 50 == 0:
                            print('Iteration {}/{}: d_loss_a={:.8f}, d_loss_b={:.8f}, g_loss={:.8f}, g_loss_fake={:.8f}, '
                                  'g_loss_gan={:.8f}\n'.format(iter, _train_iteration, loss_total[
                                epoch, 0], loss_total[epoch, 1], loss_total[epoch, 2], loss_total[epoch, 3], loss_total[epoch, 4]))

                    print(
                        'Epoch {} results: d_loss_a={:.8f}, d_loss_b={:.8f}, g_loss={:.8f}, g_loss_fake={:.8f}, g_loss_gan={:.8f}\n'.format(
                            (epoch + 1), loss_total[epoch, 0], loss_total[epoch, 1], loss_total[epoch, 2],
                            loss_total[epoch, 3], loss_total[epoch, 4]))
                    print('Starting validation\n')
                    for iter in range(_valid_iteration):
                        value = sess.run(valid_one_element)
                        g_loss = sess.run(self.gen_loss, feed_dict={self.input_a: value[0], self.input_b: value[1],
                                                self.is_training: True})

                        g_loss_valid[iter] = g_loss
                    loss_total[epoch, 5] = np.mean(g_loss_valid[np.where(g_loss_valid)])
                    print('Validation: g_loss={:.8f}\n'.format(loss_total[epoch, 5]))

                    np.savetxt(os.path.join(self.params.save_dir, 'loss.txt'), loss_total)

                    if epoch == 0:
                        path_tmp = os.path.join(self.params.save_dir, 'CRAN_model_val_loss')
                        if os.path.exists(path_tmp):
                            shutil.rmtree(path_tmp)
                        os.makedirs(path_tmp)
                        print('Save CRAN_model_val_loss\n')
                        self.saver.save(sess, path_tmp + '/model.ckpt')
                        print('Complete CRAN_model_val_loss\n')
                    else:
                        g_loss_valid_min = np.amin(loss_total[:epoch, -1])
                        if g_loss_valid_min > loss_total[epoch, -1]:
                            path_tmp = os.path.join(self.params.save_dir, 'CRAN_model_val_loss')
                            if os.path.exists(path_tmp):
                                shutil.rmtree(path_tmp)
                            os.makedirs(path_tmp)
                            print('Save CRAN_model_val_loss\n')
                            self.saver.save(sess, path_tmp + '/model.ckpt')
                            print('Complete CRAN_model_val_loss\n')
                    # save checkpoint
                    if epoch >= 0:
                        num = epoch % 3
                        path = os.path.join(self.params.save_dir, 'ckpt%d' % num)
                        if os.path.exists(path):
                            shutil.rmtree(path)
                        os.makedirs(path)
                        self.saver.save(sess, os.path.join(path, 'model.ckpt'))

                    # early stop
                    # if epoch >= 3:
                    #     if loss_total[epoch, 5] > loss_total[epoch - 1, 5]:
                    #         count = count + 1
                    #     else:
                    #         count = 0
                    #     if count >= 3:
                    #         np.savetxt(os.path.join(self.params.save_dir, 'loss.txt'), loss_total)
                    #         break
                    # early stop
                    if epoch >= 3:
                        g_loss_test = loss_total[:epoch, 5]
                        tmp = np.amin(g_loss_test)
                        if tmp < loss_total[epoch, 5]:
                            count = count + 1
                        else:
                            count = 0
                        if count >= 3:
                            np.savetxt(os.path.join(self.params.save_dir, 'loss.txt'), loss_total)
                            break

                    time_cost = time.time() - time_start
                    print('time: {}'.format(time_cost))

    def test(self, SAR2OPT_checkpoint_path=None):
        self.model_setup()
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            self.fake_ab = self.build_generator(self.input_a, self.params.OPT_IMAGE_CHANNELS, self.is_training, 'CRAN1_G', False)
        # with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        #     predict_real_b = CRAN_discriminator(self.input_b, self.params, 'CRAN_D', False)
        #     predict_fake_b = CRAN_discriminator(self.fake_b, self.params, 'CRAN_D', True)

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.saver = tf.train.Saver()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as self.sess:
            self.sess.run(init)

            if SAR2OPT_checkpoint_path is not None:
                ckpt = tf.train.get_checkpoint_state(SAR2OPT_checkpoint_path)
                if ckpt:
                    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                    print('Complete loading CRAN model!\n ')

            train_sar_save, train_opt_save, train_predict_opt_save = self.save_sample_result(self.params.num_save,
                                        os.sep.join([self.params.sample_dir, 'train/sar']),
                                        os.sep.join([self.params.sample_dir, 'train/opt']), self.sess)
            # valid_sar_save, valid_opt_save, valid_predict_opt_save = self.save_sample_result(self.params.num_save,
            #                             os.sep.join([self.params.sample_dir, 'valid/sar']),
            #                             os.sep.join([self.params.sample_dir,'valid/opt']), self.sess)
            test_sar_save, test_opt_save, test_predict_opt_save = self.save_sample_result(self.params.num_save,
                                        os.sep.join([self.params.sample_dir, 'test/sar']),
                                        os.sep.join([self.params.sample_dir, 'test/opt']), self.sess)

            f = h5py.File(os.path.join(self.params.save_dir, 'sar2opt.h5'), 'w')
            f.create_dataset('train_sar_save', data=train_sar_save)
            f.create_dataset('train_opt_save', data=train_opt_save)
            f.create_dataset('train_predict_opt_save', data=train_predict_opt_save)
            # f.create_dataset('valid_sar_save', data=valid_sar_save)
            # f.create_dataset('valid_opt_save', data=valid_opt_save)
            # f.create_dataset('valid_predict_opt_save', data=valid_predict_opt_save)
            f.create_dataset('test_sar_save', data=test_sar_save)
            f.create_dataset('test_opt_save', data=test_opt_save)
            f.create_dataset('test_predict_opt_save', data=test_predict_opt_save)
            f.close()

    def save_sample_result(self, num_save, sample_dir1, sample_dir2, sess):
        filenames = os.listdir(sample_dir1)
        num_files = len(filenames)
        index = random.sample(list(np.arange(num_files)), num_save)
        data_save1 = np.zeros((num_save, self.params.IMG_HEIGHT, self.params.IMG_WIDTH, self.params.SAR_IMAGE_CHANNELS))
        data_save2 = np.zeros((num_save, self.params.IMG_HEIGHT, self.params.IMG_WIDTH, self.params.OPT_IMAGE_CHANNELS))
        predict_data_save = np.zeros((num_save, self.params.IMG_HEIGHT, self.params.IMG_WIDTH, self.params.OPT_IMAGE_CHANNELS))
        for i in range(num_save):
            data_save1[i] = cv2.imread(os.sep.join([sample_dir1, filenames[index[i]]])) / 127.5 - 1.
            data_save2[i] = cv2.cvtColor(cv2.imread(os.sep.join([sample_dir2, filenames[index[i]]])), cv2.COLOR_BGR2RGB)
            predict_data_save[i] = np.squeeze(sess.run(self.fake_ab, feed_dict={self.input_a: data_save1[i][np.newaxis], self.is_training: True}))
        data_save1, predict_data_save = _norm_image_inv(data_save1), _norm_image_inv(predict_data_save)
        return data_save1, data_save2, predict_data_save


if __name__ == '__main__':
    model = create_model(params)
    # model.train()
    model.test("../results_GF3_new/ckpt0")