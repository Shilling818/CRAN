from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.preprocessing.image import img_to_array, load_img
import cv2
import os
import numpy as np
from tensorflow.python.keras.utils import multi_gpu_model
import scipy.io as io
from deeplab_weights import Deeplabv3
import argparse
import random
import h5py
from dataset2 import *
import cv2
import shutil
import tensorflow as tf
from model import *


os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2, 3"


parser = argparse.ArgumentParser(description='SAR Segmentation')
parser.add_argument('--name', type=str, default='SAR Segmentation', help='Experiment Name')
parser.add_argument('--sample_dir', type=str, default='../dataset/256_building_aug_new')
parser.add_argument('--save_dir', type=str, default='../results_my_opt2map_256_building_aug_new', help='directory to save results')
parser.add_argument('--pretrained_weights_dir', type=str, default='../pretrained_weights/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5')
parser.add_argument('--IMG_HEIGHT', type=int, default=256)
parser.add_argument('--IMG_WIDTH', type=int, default=256)
parser.add_argument('--MAP_IMAGE_CHANNELS', type=int, default=5)
parser.add_argument('--SAR_IMAGE_CHANNELS', type=int, default=3)
parser.add_argument('--OPT_IMAGE_CHANNELS', type=int, default=3)
# building, vegetation, water, road, bare_land
parser.add_argument('--color_map', type=list, default=[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 0, 0]])
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--max_epoch', type=int, default=1, help='max epochs to train')
parser.add_argument('--num_save', type=int, default=15)
parser.add_argument('--ngf', type=int, default=50)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--l1_weight', default=100.0, type=float)
parser.add_argument('--gan_weight', default=5.0, type=float)
parser.add_argument('--beta1', default=0.5, type=float, help='''Adam Optimizer''')
params = parser.parse_args()


def one_hot_lab(labels):
    lab_nd = np.zeros([params.IMG_HEIGHT, params.IMG_WIDTH, params.MAP_IMAGE_CHANNELS])
    # building, vegetation, water, road, bare_land
    color_map = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 0, 0]]
    for i in range(params.MAP_IMAGE_CHANNELS):
        lab_nd[:, :, i] = np.floor(np.array(labels == color_map[i], dtype='uint8').sum(axis=2) / 3)
    return lab_nd


def one_hot_lab_inv(labels):
    lab_nd = np.zeros([params.IMG_HEIGHT, params.IMG_WIDTH, 3])
    for i in range(params.MAP_IMAGE_CHANNELS):
        tmp = np.array(params.color_map[i]).reshape((1, 1, 3)).repeat(params.IMG_HEIGHT, axis=0).repeat(params.IMG_WIDTH,
                                                                                                 axis=1)
        lab_nd += np.multiply(np.array(labels == i)[:, :, np.newaxis].repeat(3, axis=2), tmp)
    return lab_nd


def sample_logs(hist, string):
    out = hist.history[string]
    out = np.array(out).reshape((1, len(out)))
    return out

def save_logs(hist):
    out = np.concatenate([sample_logs(hist, 'acc'), sample_logs(hist, 'loss'), sample_logs(hist, 'val_acc'),
                          sample_logs(hist, 'val_loss')], axis=0)
    np.savetxt(os.path.join(params.save_dir, 'logs.txt'), out)


class model_OPT2MAP():
    def __init__(self): pass
        # super(model_OPT2MAP, self).__init__()

    def save_sample_result(self, num_save, sample_dir1, sample_dir2):
        filenames = os.listdir(sample_dir1)
        num_files = len(filenames)
        index = random.sample(list(np.arange(num_files)), num_save)
        data_save1 = np.zeros((num_save, params.IMG_HEIGHT, params.IMG_WIDTH, params.OPT_IMAGE_CHANNELS))
        data_save2 = np.zeros((num_save, params.IMG_HEIGHT, params.IMG_WIDTH, 3))
        predict_data_save = np.zeros((num_save, params.IMG_HEIGHT, params.IMG_WIDTH, 3))
        for i in range(num_save):
            data_save1[i] = cv2.imread(os.sep.join([sample_dir1, filenames[index[i]]])) / 127.5 - 1.
            data_save2[i] = cv2.cvtColor(cv2.imread(os.sep.join([sample_dir2, filenames[index[i]]])), cv2.COLOR_BGR2RGB)
        predict_result = self.model.predict(data_save1, batch_size=1, verbose=1)

        for i in range(num_save):
            predict_data_save[i] = one_hot_lab_inv(predict_result[i].argmax(axis=2))
        data_save1 = (data_save1 + 1.) * 127.5
        return data_save1, data_save2, predict_data_save

    def train(self):
        num_train_dataset = len(os.listdir(os.sep.join([params.sample_dir, 'train/opt'])))
        num_valid_dataset = len(os.listdir(os.sep.join([params.sample_dir, 'valid/opt'])))

        print('Loading data...')
        train_dataset = dataset_OPT2MAP(os.sep.join([params.sample_dir, 'train/opt']),
                                os.sep.join([params.sample_dir, 'train/map']), params, num_train_dataset)
        valid_dataset = dataset_OPT2MAP(os.sep.join([params.sample_dir, 'valid/opt']),
                                os.sep.join([params.sample_dir, 'valid/map']), params, num_valid_dataset)
        print('Loading data done!')

        if os.path.exists(params.save_dir):
            shutil.rmtree(params.save_dir)
        os.makedirs(params.save_dir)

        self.model = Deeplabv3(input_shape=(params.IMG_HEIGHT, params.IMG_WIDTH, params.OPT_IMAGE_CHANNELS),
                          classes=params.MAP_IMAGE_CHANNELS)  # 256, 256, 3 -> 256 256 5
        # model = multi_gpu_model(model_single, gpus=4)

        self.model.load_weights(params.pretrained_weights_dir, by_name=True)
        self.model.summary()
        model_checkpoint = ModelCheckpoint(os.path.join(params.save_dir, 'deeplabv3_val_acc.hdf5'), monitor='val_acc',
                                           verbose=1,
                                           save_best_only=True, save_weights_only=True)
        model_checkpoint1 = ModelCheckpoint(os.path.join(params.save_dir, 'deeplabv3_train_loss.hdf5'), monitor='loss',
                                            verbose=1,
                                            save_best_only=True, save_weights_only=True)
        model_checkpoint2 = ModelCheckpoint(os.path.join(params.save_dir, 'deeplabv3_val_loss.hdf5'), monitor='val_loss',
                                            verbose=1,
                                            save_best_only=True, save_weights_only=True)
        model_earlystop = EarlyStopping(patience=3, monitor='val_acc', verbose=2)
        print('Fitting model...')
        self.model.compile(optimizer=Adam(lr=params.learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        hist = self.model.fit(train_dataset, epochs=params.max_epoch, validation_data=valid_dataset, steps_per_epoch=num_train_dataset // params.batch_size,
                  validation_steps=num_valid_dataset // params.batch_size, callbacks=[model_checkpoint, model_earlystop, model_checkpoint1, model_checkpoint2])
        save_logs(hist)

    def test(self, pretrained_path=None):
        self.model = Deeplabv3(input_shape=(params.IMG_HEIGHT, params.IMG_WIDTH, params.OPT_IMAGE_CHANNELS),
                                 classes=params.MAP_IMAGE_CHANNELS)  # 256, 256, 3 -> 256 256 5
        # model = multi_gpu_model(model_single, gpus=4)
        if pretrained_path is not None:
            self.model.load_weights(pretrained_path, by_name=True)
        self.model.summary()
        self.model.compile(optimizer=Adam(lr=params.learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        # randomly sample,and predict the results
        train_opt_save, train_map_save, train_predict_map_save = self.save_sample_result(params.num_save,
                        os.sep.join([params.sample_dir, 'train/opt']), os.sep.join([params.sample_dir, 'train/map']))
        valid_opt_save, valid_map_save, valid_predict_map_save = self.save_sample_result(params.num_save,
                        os.sep.join([params.sample_dir, 'valid/opt']), os.sep.join([params.sample_dir, 'valid/map']))
        test_opt_save, test_map_save, test_predict_map_save = self.save_sample_result(params.num_save,
                        os.sep.join([params.sample_dir, 'test/opt']), os.sep.join([params.sample_dir, 'test/map']))
        f = h5py.File(os.path.join(params.save_dir, 'sar2opt.h5'), 'w')
        f.create_dataset('train_opt_save', data=train_opt_save)
        f.create_dataset('train_map_save', data=train_map_save)
        f.create_dataset('train_predict_map_save', data=train_predict_map_save)
        f.create_dataset('valid_opt_save', data=valid_opt_save)
        f.create_dataset('valid_map_save', data=valid_map_save)
        f.create_dataset('valid_predict_map_save', data=valid_predict_map_save)
        f.create_dataset('test_opt_save', data=test_opt_save)
        f.create_dataset('test_map_save', data=test_map_save)
        f.create_dataset('test_predict_map_save', data=test_predict_map_save)
        f.close()

    def evaluate(self, pretrained_path):
        num_test_dataset = len(os.listdir(os.sep.join([params.sample_dir, 'test_aug/opt'])))
        print('Loading data...')
        test_dataset = dataset_OPT2MAP(os.sep.join([params.sample_dir, 'test_aug/opt']),
                                os.sep.join([params.sample_dir, 'test_aug/map']), params, num_test_dataset)
        print('Loading data done!')
        self.model = Deeplabv3(input_shape=(params.IMG_HEIGHT, params.IMG_WIDTH, params.OPT_IMAGE_CHANNELS),
                               classes=params.MAP_IMAGE_CHANNELS)  # 256, 256, 3 -> 256 256 5
        # model = multi_gpu_model(model_single, gpus=4)
        if pretrained_path is not None:
            self.model.load_weights(pretrained_path, by_name=True)
            print('Loading model done!')
        self.model.summary()
        self.model.compile(optimizer=Adam(lr=params.learning_rate), loss='categorical_crossentropy',
                           metrics=['accuracy'])
        scores = self.model.evaluate(test_dataset, steps=num_test_dataset // params.batch_size)
        print(self.model.metrics_names)
        print(scores)

    def save_sample_result_whole(self, sample_dir):
        filenames = os.listdir(sample_dir)
        filenames.sort(key=lambda x: int(x[:-4]))
        num_files = len(filenames)
        data_tmp = np.zeros((num_files, params.IMG_HEIGHT, params.IMG_WIDTH, params.SAR_IMAGE_CHANNELS))
        data_save = np.zeros((num_files, params.IMG_HEIGHT, params.IMG_WIDTH, 3))
        for i in range(num_save):
            data_tmp[i] = cv2.imread(os.sep.join([sample_dir, filenames[i]])) / 127.5 - 1.
        predict_result = self.model.predict(data_tmp, batch_size=1, verbose=1)
        for i in range(num_files):
            data_save[i] = one_hot_lab_inv(predict_result[i].argmax(axis=2))
        return data_save

    def test_whole(self, pretrained_path):
        self.model = Deeplabv3(input_shape=(params.IMG_HEIGHT, params.IMG_WIDTH, params.SAR_IMAGE_CHANNELS),
                               classes=params.MAP_IMAGE_CHANNELS)  # 256, 256, 3 -> 256 256 5
        # model = multi_gpu_model(model_single, gpus=4)
        if pretrained_path is not None:
            self.model.load_weights(pretrained_path, by_name=True)
        self.model.summary()
        self.model.compile(optimizer=Adam(lr=params.learning_rate), loss='categorical_crossentropy',
                           metrics=['accuracy'])
        train_predict_map_save = self.save_sample_result_whole(os.sep.join([params.sample_dir, 'train/opt']))
        valid_predict_map_save = self.save_sample_result_whole(os.sep.join([params.sample_dir, 'valid/opt']))
        test_predict_map_save = self.save_sample_result_whole(os.sep.join([params.sample_dir, 'test/opt']))
        f = h5py.File(os.path.join(params.save_dir, 'data.h5'), 'w')
        f.create_dataset('train_predict_map_save', data=train_predict_map_save)
        f.create_dataset('valid_predict_map_save', data=valid_predict_map_save)
        f.create_dataset('test_predict_map_save', data=test_predict_map_save)
        f.close()


if __name__ == '__main__':
    # train SAR2OPT
    # model = model_SAR2OPT(params)
    # model.train()
    # model.test(SAR2OPT_checkpoint_path='../results_sar2opt/CRAN_model_val_loss')

    # train OPT2MAP
    model = model_OPT2MAP()
    # model.train()
    # model.test('../results_my_256_building_aug_new/deeplabv3_val_loss.hdf5')
    model.evaluate('../results_my_opt2map_256_building_aug_new/deeplabv3_val_acc.hdf5')

    # train SAR2OPT2MAP
    # model_combined = model_SAR2OPT2MAP()
    # model.train(params, SAR2OPT_checkpoint_path=, OPT2MAP_checkpoint_path=)
