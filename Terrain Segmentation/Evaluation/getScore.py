import numpy as np
import os
import cv2
import argparse


parser = argparse.ArgumentParser(description='Segmentation Scores')
parser.add_argument('--name', type=str, default='Segmentation Scores', help='Experiment Name')
parser.add_argument('--IMG_HEIGHT', type=int, default=256)
parser.add_argument('--IMG_WIDTH', type=int, default=256)
parser.add_argument('--MAP_IMAGE_CHANNELS', type=int, default=5)
parser.add_argument('--SAR_IMAGE_CHANNELS', type=int, default=3)
# building, vegetation, water, road, bare_land
parser.add_argument('--color_map', type=list, default=[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 0, 0]])
params = parser.parse_args()


def fast_hist(a, b, n):
    # print('saving')
    # sio.savemat('/tmp/fcn_debug/xx.mat', {'a':a, 'b':b, 'n':n})

    k = np.where((a >= 0) & (a < n))[0]
    bc = np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2)
    if len(bc) != n ** 2:
        # ignore this example if dimension mismatch
        return 0
    return bc.reshape(n, n)


def get_scores(hist):
    # Mean pixel accuracy
    acc = np.diag(hist).sum() / (hist.sum() + 1e-12)

    # Per class accuracy
    cl_acc = np.diag(hist) / (hist.sum(1) + 1e-12)

    # Per class IoU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-12)

    return acc, np.nanmean(cl_acc), np.nanmean(iu), cl_acc, iu


def one_hot_lab(labels):
    lab_nd = np.zeros([params.IMG_HEIGHT, params.IMG_WIDTH, params.MAP_IMAGE_CHANNELS])
    # building, vegetation, water, road, bare_land
    color_map = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 0, 0]]
    for i in range(params.MAP_IMAGE_CHANNELS):
        lab_nd[:, :, i] = np.floor(np.array(labels == color_map[i], dtype='uint8').sum(axis=2) / 3) * (i + 1)
    lab_nd = lab_nd.sum(axis=2) - 1.
    lab_nd = np.array(lab_nd, dtype='int64')
    return lab_nd


def main():
    # map path of ground Truth
    gt_samples = ''
    # map path of predicted images
    predict_samples = ''
    filenames = os.listdir(gt_samples)
    filenames.sort(key=lambda x: int(x[:-4]))
    num_files = len(filenames)

    hist_perframe = np.zeros((params.MAP_IMAGE_CHANNELS, params.MAP_IMAGE_CHANNELS))
    for i in range(num_files):
        gt = one_hot_lab(cv2.cvtColor(cv2.imread(os.sep.join([gt_samples, filenames[i]])), cv2.COLOR_BGR2RGB))
        predict = one_hot_lab(cv2.cvtColor(cv2.imread(os.sep.join([predict_samples, filenames[i]])), cv2.COLOR_BGR2RGB))
        hist_perframe += fast_hist(gt.flatten(), predict.flatten(), params.MAP_IMAGE_CHANNELS)
    mean_pixel_acc, mean_class_acc, mean_class_iou, per_class_acc, per_class_iou = get_scores(hist_perframe)
    print('mean_pixel_acc:',  mean_pixel_acc)
    print('mean_class_acc: ', mean_class_acc)
    print('mean_class_iou: ', mean_class_iou)
    print('per_class_acc: ', per_class_acc)
    print('per_class_iou: ', per_class_iou)


if __name__ == '__main__':
    main()
