import tensorflow as tf
import numpy as np
import os
from tensorflow.python.keras.preprocessing.image import img_to_array, load_img


def load_sample(sample_dir1, sample_dir2):
    print("Loading sample dataset...")
    filename1_path = []
    filename2_path = []
    filenames = os.listdir(sample_dir1)
    filenames.sort(key=lambda x: int(x[:-4]))
    for filename in filenames:
        filename1_path.append(os.sep.join([sample_dir1, filename]))
        filename2_path.append(os.sep.join([sample_dir2, filename]))
    return filename1_path, filename2_path


def load_sample3(sample_dir1, sample_dir2, sample_dir3):
    print("Loading sample dataset...")
    filename1_path = []
    filename2_path = []
    filename3_path = []
    filenames = os.listdir(sample_dir1)
    filenames.sort(key=lambda x: int(x[:-4]))
    for filename in filenames:
        filename1_path.append(os.sep.join([sample_dir1, filename]))
        filename2_path.append(os.sep.join([sample_dir2, filename]))
        filename3_path.append(os.sep.join([sample_dir3, filename]))
    return filename1_path, filename2_path, filename3_path


def _norm_image(image):
    return image / 127.5 - 1


def one_hot_lab(labels, params):
    for i in range(params.MAP_IMAGE_CHANNELS):
        tmp = tf.equal(labels, tf.cast(tf.tile(tf.reshape(tf.constant(params.color_map[i]), [1, 1, 3]), [params.IMG_HEIGHT, params.IMG_WIDTH, 1]), dtype=tf.float32))
        tmp = tf.cast(tmp, dtype=tf.float32)
        tmp = tf.expand_dims(tf.math.floor(tf.reduce_sum(tmp, axis=2) / 3), axis=2)
        if i == 0:
            lab_nd = tmp
        else:
            lab_nd = tf.concat((lab_nd, tmp), axis=2)
    lab_nd = tf.reshape(lab_nd, [params.IMG_HEIGHT, params.IMG_WIDTH, params.MAP_IMAGE_CHANNELS])
    return lab_nd


def dataset_OPT2MAP(sample_dir1, sample_dir2, params, shuffle):
    filename_path1, filename_path2 = load_sample(sample_dir1, sample_dir2)

    def _parseone_OPT2MAP(filename1, filename2):
        image_string = tf.read_file(filename1)
        image_decoded = tf.image.decode_image(image_string)
        image_decoded = tf.cast(image_decoded, dtype=tf.float32)
        image_decoded = tf.reshape(_norm_image(image_decoded), [params.IMG_HEIGHT, params.IMG_WIDTH, params.OPT_IMAGE_CHANNELS])

        image_string = tf.read_file(filename2)
        image_decoded2 = tf.image.decode_image(image_string)
        image_decoded2 = tf.cast(image_decoded2, dtype=tf.float32)
        image_decoded2 = tf.reshape(one_hot_lab(image_decoded2, params), [params.IMG_HEIGHT, params.IMG_WIDTH, params.MAP_IMAGE_CHANNELS])
        return image_decoded, image_decoded2

    dataset = tf.data.Dataset.from_tensor_slices((filename_path1, filename_path2))
    dataset = dataset.map(_parseone_OPT2MAP)
    if shuffle:
        dataset = dataset.shuffle(params.batch_size)
    dataset = dataset.batch(params.batch_size, drop_remainder=True)
    dataset = dataset.repeat(params.max_epoch)
    return dataset


def dataset_SAR2OPT(sample_dir1, sample_dir2, params, shuffle_num):
    filename_path1, filename_path2 = load_sample(sample_dir1, sample_dir2)

    def _parseone_SAR2OPT(filename1, filename2):
        image_string = tf.read_file(filename1)
        image_decoded = tf.image.decode_image(image_string)
        image_decoded = tf.cast(image_decoded, dtype=tf.float32)
        # if tf.shape(image_decoded)[0] == 2:
        image_decoded = tf.expand_dims(image_decoded, axis=2)
        image_decoded = tf.concat((image_decoded, image_decoded, image_decoded), axis=2)
        image_decoded = tf.reshape(_norm_image(image_decoded), [params.IMG_HEIGHT, params.IMG_WIDTH, params.SAR_IMAGE_CHANNELS])

        image_string = tf.read_file(filename2)
        image_decoded2 = tf.image.decode_image(image_string)
        image_decoded2 = tf.cast(image_decoded2, dtype=tf.float32)
        image_decoded2 = tf.reshape(_norm_image(image_decoded2), [params.IMG_HEIGHT, params.IMG_WIDTH, params.OPT_IMAGE_CHANNELS])
        return image_decoded, image_decoded2

    dataset = tf.data.Dataset.from_tensor_slices((filename_path1, filename_path2))
    dataset = dataset.map(_parseone_SAR2OPT)
    dataset = dataset.shuffle(shuffle_num)
    dataset = dataset.batch(params.batch_size, drop_remainder=True)
    dataset = dataset.repeat(params.max_epoch)
    return dataset


def dataset_SAR2OPT2MAP(sample_dir1, sample_dir2, sample_dir3, params, shuffle):
    filename_path1, filename_path2, filename_path3 = load_sample3(sample_dir1, sample_dir2, sample_dir3)

    def _parseone_SAR2OPT2MAP(filename1, filename2, filename3):
        image_string = tf.read_file(filename1)
        image_decoded = tf.image.decode_image(image_string)
        image_decoded = tf.cast(image_decoded, dtype=tf.float32)
        image_decoded = tf.reshape(_norm_image(image_decoded), [params.IMG_HEIGHT, params.IMG_WIDTH, params.SAR_IMAGE_CHANNELS])

        image_string = tf.read_file(filename2)
        image_decoded2 = tf.image.decode_image(image_string)
        image_decoded2 = tf.cast(image_decoded2, dtype=tf.float32)
        image_decoded2 = tf.reshape(_norm_image(image_decoded2), [params.IMG_HEIGHT, params.IMG_WIDTH, params.OPT_IMAGE_CHANNELS])

        image_string = tf.read_file(filename3)
        image_decoded3 = tf.image.decode_image(image_string)
        image_decoded3 = tf.cast(image_decoded3, dtype=tf.float32)
        image_decoded3 = tf.reshape(one_hot_lab(image_decoded3, params), [params.IMG_HEIGHT, params.IMG_WIDTH, params.MAP_IMAGE_CHANNELS])
        return image_decoded, image_decoded2, image_decoded3

    dataset = tf.data.Dataset.from_tensor_slices((filename_path1, filename_path2, filename_path3))
    dataset = dataset.map(_parseone_SAR2OPT2MAP)
    if shuffle:
        dataset = dataset.shuffle(params.batch_size)
    dataset = dataset.batch(params.batch_size, drop_remainder=True)
    dataset = dataset.repeat(params.max_epoch)
    return dataset


def getone(dataset):
    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()
    return one_element