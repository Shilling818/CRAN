# -*- coding: UTF-8 -*-
import tensorflow as tf

EPS = 1e-12


def gan_loss_discriminator(predict_real, predict_fake):
    return tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))


def gan_loss_generator(predict_fake):
    return tf.reduce_mean(-tf.log(predict_fake + EPS))


def L1_loss(real, fake):
    return tf.reduce_mean(tf.abs(real - fake))


def wgan_loss_discriminator(predict_real, predict_fake):
    return tf.abs(- tf.reduce_mean(predict_real) + tf.reduce_mean(predict_fake))


def wgan_loss_generator(predict_fake):
    return tf.abs(- tf.reduce_mean(predict_fake))


def lsgan_loss_discriminator(predict_real, predict_fake):
    return (tf.reduce_mean(tf.squared_difference(predict_real, 1)) +
            tf.reduce_mean(tf.squared_difference(predict_fake, 0))) * 0.5


def lsgan_loss_generator(predict_fake):
    return tf.reduce_mean(tf.squared_difference(predict_fake, 1))

