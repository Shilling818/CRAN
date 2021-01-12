import argparse
import numpy as np
import cv2
import os
import random
import shutil
import scipy.io

parser = argparse.ArgumentParser(description='SAR Segmentation')
parser.add_argument('--name', type=str, default='SAR Segmentation', help='Experiment Name')
parser.add_argument('--IMG_HEIGHT', type=int, default=256)
parser.add_argument('--IMG_WIDTH', type=int, default=256)
parser.add_argument('--color_map', type=list, default=[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 0, 0]])
parser.add_argument('--num_color_map', type=int, default=5)
parser.add_argument('--ratio_train_test', type=float, default=0.15)
params = parser.parse_args()


# cut a large image into patches
def slice_into_patch(img, save_dir, save_index):
    h, w = img.shape[:2]
    ratio1, ratio2 = h // params.IMG_HEIGHT, w // params.IMG_WIDTH
    # img_tmp = np.zeros((ratio1 * ratio2, params.IMG_HEIGHT, params.IMG_WIDTH))
    for i in range(ratio1):
        for j in range(ratio2):
            img_tmp = img[(i * params.IMG_HEIGHT): ((i + 1) * params.IMG_HEIGHT),
                      (j * params.IMG_WIDTH): ((j + 1) * params.IMG_WIDTH)]
            cv2.imwrite(os.sep.join([save_dir, '%.8d.png' % save_index]), cv2.cvtColor(img_tmp, cv2.COLOR_RGB2BGR))
            save_index += 1
    return save_index


# cut many large images into dataset
def slice_dataset(sample_dirs, save_dirs):
    num_sample_dirs = len(sample_dirs)
    filenames = os.listdir(sample_dirs[0])
    save_index = [0, 0, 0]

    for num in range(num_sample_dirs):
        if os.path.exists(save_dirs[num]):
            shutil.rmtree(save_dirs[num])
        os.makedirs(save_dirs[num])

    for name in filenames:
        for num in range(num_sample_dirs):
            name1 = name
            if num == 1:
                name1 = name1.replace('Label', 'SAR')
            elif num == 2:
                name1 = name1.replace('Label', 'Optical')

            save_index[num] = slice_into_patch(cv2.cvtColor(cv2.imread(os.sep.join([sample_dirs[num], name1])), cv2.COLOR_BGR2RGB), save_dirs[num],
                                               save_index[num])
        assert len(set(save_index)) == 1


def count_per_img(img):
    freq = np.zeros((params.num_color_map))
    for i in range(params.num_color_map):
        freq[i] = np.floor(np.array(img == params.color_map[i], dtype='uint8').sum(axis=2) / 3).sum()
    label = freq.argmax()
    return freq, label


def count_dataset(sample_dir):
    num_files = len(os.listdir(sample_dir))
    freqs = np.zeros((num_files, params.num_color_map))
    labels = np.zeros((num_files, 1))
    indexs = []
    for i in range(num_files):
        path = os.sep.join([sample_dir, '%.8d.png' % i])
        # path = os.sep.join([sample_dir, '%.5d.png' % (i + 1)])
        assert os.path.exists(path)
        freqs[i], labels[i] = count_per_img(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
    for i in range(params.num_color_map):
        indexs.append(np.argwhere(np.array(labels == i))[:, 0])
    return freqs, labels, indexs


def dataset_balance_index(indexs):
    freq_labels = np.zeros((params.num_color_map))
    for i in range(params.num_color_map):
        freq_labels[i] = indexs[i].shape[0]
    max_freq = freq_labels.max()
    new_indexs = []
    for i in range(params.num_color_map):
        if freq_labels[i] > 0:
            tmp = indexs[i]
            num = int(max_freq // freq_labels[i]) - 1
            for j in range(num):
                tmp = np.concatenate((tmp, indexs[i]), axis=0)
            num = int(max_freq % freq_labels[i])
            if num > 0:
                tmp = np.concatenate((np.array(random.sample(list(indexs[i]), num)), tmp), axis=0)
            assert tmp.shape[0] == max_freq
            new_indexs.append(tmp)
    return freq_labels, new_indexs


def dataset_balance_pixel(freqs):
    num_freqs = freqs.shape[0]
    freqs_class = freqs.sum(axis=0)
    max_freqs_class, max_freqs_class_index = freqs_class.max(), freqs_class.argmax()
    new_indexs = []
    # for i in range(params.num_color_map):
    for i in range(1):
        if i == max_freqs_class_index:
            continue
        diff = max_freqs_class - freqs_class[i]
        freqs_current = freqs[:, i]
        freqs_current_sort, freqs_current_sort_index = np.sort(freqs_current, axis=0)[::-1], np.argsort(freqs_current, axis=0)[::-1]
        # interval = int(num_freqs * 0.1)
        # freqs_current_sort = freqs_current_sort[:interval]
        # freqs_current_sort_index = freqs_current_sort_index[:interval]
        index_selected = np.where(freqs_current_sort >= 32768)
        freqs_current_sort = freqs_current_sort[index_selected]
        freqs_current_sort_index = freqs_current_sort_index[index_selected]
        interval = len(index_selected[0])
        if interval == 0:
            continue
        index = []
        while diff >= 0:
            count = random.randint(0, interval - 1)
            diff -= freqs_current_sort[count]
            index_tmp = freqs_current_sort_index[count]
            freqs_class += freqs[index_tmp, :]
            index.append(index_tmp)
        new_indexs += index
    return new_indexs, freqs_class


def dataset_copy(new_indexs, sample_dirs, save_dirs):
    num_sample_dirs = len(sample_dirs)

    for num in range(num_sample_dirs):
        if os.path.exists(save_dirs[num]):
            shutil.rmtree(save_dirs[num])
        os.makedirs(save_dirs[num])

    print('Create New Dataset...\n')
    for i in range(len(new_indexs)):
        num_sub_indexs = new_indexs[i].shape[0]
        for j in range(num_sub_indexs):
            count = i * num_sub_indexs + j
            for k in range(num_sample_dirs):
                shutil.copy(os.sep.join([sample_dirs[k], '%.8d.png' % new_indexs[i][j]]),
                            os.sep.join([save_dirs[k], '%.8d.png' % count]))
    print('Complete Dataset Creation!')


def dataset_copy2(new_indexs, sample_dirs, save_dirs):
    num_sample_dirs = len(sample_dirs)

    for num in range(num_sample_dirs):
        if os.path.exists(save_dirs[num]):
            shutil.rmtree(save_dirs[num])
        os.makedirs(save_dirs[num])

    print('Create New Dataset...\n')
    for i in range(len(new_indexs)):
        flag1, flag2 = random.random() > 0.5, random.random() > 0.5
        for k in range(num_sample_dirs):
            path = os.sep.join([sample_dirs[k], '%.8d.png' % new_indexs[i]])
            tmp = cv2.imread(path)
            if flag1:
                tmp = np.fliplr(tmp)
            if flag2:
                tmp = np.flipud(tmp)
            cv2.imwrite(os.sep.join([save_dirs[k], '%.8d.png' % i]), tmp)
    print('Complete Dataset Creation!')


def split_train_valid_test_dataset(indexs):
    train_indexs = []
    test_indexs = []
    valid_indexs = []
    for i in range(params.num_color_map):
        num_sub_valid_test_indexs = int(np.ceil(indexs[i].shape[0] * params.ratio_train_test * 2))
        tmp = np.array(random.sample(list(indexs[i]), num_sub_valid_test_indexs))
        test_tmp = np.array(random.sample(list(tmp), num_sub_valid_test_indexs // 2))
        test_indexs.append(test_tmp)
        train_indexs.append(np.array(list(set(indexs[i]) ^ set(tmp))))
        valid_indexs.append(np.array(list(set(tmp) ^ set(test_tmp))))
    return train_indexs, valid_indexs, test_indexs


def split_train_test_dataset(indexs):
    # train_indexs = []
    # test_indexs = []
    num_sub_test_indexs = int(np.ceil(len(indexs) * 0.3))
    test_indexs = np.array(random.sample(list(indexs), num_sub_test_indexs))
    train_indexs = np.array(list(set(indexs) ^ set(test_indexs)))
    return train_indexs, test_indexs


def aug_samples(indexs, sample_dirs, save_dirs, repeats):
    num_sample_dirs = len(sample_dirs)

    for num in range(num_sample_dirs):
        if os.path.exists(save_dirs[num]):
            shutil.rmtree(save_dirs[num])
        os.makedirs(save_dirs[num])

    indexs = list(indexs)
    new_indexs = indexs
    while repeats:
        new_indexs += indexs
        repeats -= 1

    print('Create New Dataset...\n')
    for i in range(len(new_indexs)):
        flag1, flag2 = random.random() > 0.5, random.random() > 0.5
        for k in range(num_sample_dirs):
            path = os.sep.join([sample_dirs[k], new_indexs[i]])
            tmp = cv2.imread(path)
            if flag1:
                tmp = np.fliplr(tmp)
            if flag2:
                tmp = np.flipud(tmp)
            cv2.imwrite(os.sep.join([save_dirs[k], '%.8d.png' % i]), tmp)
    print('Complete Dataset Creation!')


def regroup(indexs):
    num_indexs = len(indexs)
    new_indexs = []
    for i in range(num_indexs):
        new_indexs += list(indexs[i])
    return new_indexs


def regroup2(list1, list2):
    num_list2 = len(list2)
    new_list = []
    for i in range(num_list2):
        new_list.append(list1[list2[i]])
    return new_list


if __name__ == '__main__':
    # cut into 256
    sample_dirs = ['F:/EMWLAB/Img2Img/Segmentation/dataset/lab_1024',
                   'F:/EMWLAB/Img2Img/Segmentation/dataset/sar_1024',
                   'F:/EMWLAB/Img2Img/Segmentation/dataset/opt_1024']
    save_dirs = ['F:/EMWLAB/Img2Img/Segmentation/dataset/lab_256',
                 'F:/EMWLAB/Img2Img/Segmentation/dataset/sar_256',
                  'F:/EMWLAB/Img2Img/Segmentation/dataset/opt_256']
    slice_dataset(sample_dirs, save_dirs)

    # freqs, labels, indexs = count_dataset('F:/EMWLAB/Img2Img/Segmentation/dataset/256/valid/map')
    # freqs, labels, indexs = count_dataset('F:/EMWLAB/Img2Img/Segmentation/dataset/sar_lab_aug/valid/map')
    # freqs2, labels2, indexs2 = count_dataset('F:/EMWLAB/Img2Img/Segmentation/dataset/sar_lab_aug/train/map')
    # freqs, labels, indexs = count_dataset('F:/EMWLAB/Img2Img/Segmentation/dataset/256_building_aug_new/valid/map')
    # freqs2, labels2, indexs2 = count_dataset('F:/EMWLAB/Img2Img/Segmentation/dataset/256_building_aug_new/train/map')
    # freqs3, labels3, indexs3 = count_dataset('F:/EMWLAB/Img2Img/Segmentation/dataset/256_building_aug_new/test/map')
    # print(freqs.sum(axis=0))
    # print(freqs2.sum(axis=0))
    # print(freqs3.sum(axis=0))

    # freqs, labels, indexs = count_dataset('F:/EMWLAB/Img2Img/Segmentation/dataset/256_building_aug_new/test/map')
    # indexs = regroup(indexs)
    # new_indexs_test, freqs_class_test = dataset_balance_pixel(freqs)
    # print(freqs_class_test)
    # new_indexs_test += indexs
    # save_dirs = ['F:/EMWLAB/Img2Img/Segmentation/dataset/256_building_aug_new/test/map',
    #              'F:/EMWLAB/Img2Img/Segmentation/dataset/256_building_aug_new/test/sar',
    #              'F:/EMWLAB/Img2Img/Segmentation/dataset/256_building_aug_new/test/opt']
    # save_dirs3 = ['F:/EMWLAB/Img2Img/Segmentation/dataset/256_building_aug_new/test_aug/map',
    #               'F:/EMWLAB/Img2Img/Segmentation/dataset/256_building_aug_new/test_aug/sar',
    #               'F:/EMWLAB/Img2Img/Segmentation/dataset/256_building_aug_new/test_aug/opt']
    # dataset_copy2(new_indexs_test, save_dirs, save_dirs3)
    # freqs, labels, indexs = count_dataset('F:/EMWLAB/Img2Img/Segmentation/dataset/256_building_aug_new/test_aug/map')
    # print(freqs.sum(axis=0))

    # freqs, labels, indexs = count_dataset('F:/EMWLAB/Img2Img/Segmentation/dataset/lab_256')
    # train_indexs, valid_indexs, test_indexs = split_train_valid_test_dataset(indexs)
    # train_indexs, valid_indexs, test_indexs = regroup(train_indexs), regroup( valid_indexs), regroup(test_indexs)
    # new_indexs_train, freqs_class_train = dataset_balance_pixel(freqs[train_indexs])
    # new_indexs_train = regroup2(train_indexs, new_indexs_train)
    # new_indexs_train += train_indexs
    # print('freqs_class_train:', freqs_class_train)
    #
    # new_indexs_valid, freqs_class_valid = dataset_balance_pixel(freqs[valid_indexs])
    # new_indexs_valid = regroup2(valid_indexs, new_indexs_valid)
    # new_indexs_valid += valid_indexs
    # print('freqs_class_valid:', freqs_class_valid)
    #
    # save_dirs1 = ['F:/EMWLAB/Img2Img/Segmentation/dataset/256/train/map',
    #               'F:/EMWLAB/Img2Img/Segmentation/dataset/256/train/sar',
    #               'F:/EMWLAB/Img2Img/Segmentation/dataset/256/train/opt']
    # dataset_copy2(new_indexs_train, save_dirs, save_dirs1)
    #
    # save_dirs2 = ['F:/EMWLAB/Img2Img/Segmentation/dataset/256/test/map',
    #               'F:/EMWLAB/Img2Img/Segmentation/dataset/256/test/sar',
    #               'F:/EMWLAB/Img2Img/Segmentation/dataset/256/test/opt']
    # dataset_copy2(test_indexs, save_dirs, save_dirs2)
    #
    # save_dirs3 = ['F:/EMWLAB/Img2Img/Segmentation/dataset/256/valid/map',
    #               'F:/EMWLAB/Img2Img/Segmentation/dataset/256/valid/sar',
    #               'F:/EMWLAB/Img2Img/Segmentation/dataset/256/valid/opt']
    # dataset_copy2(new_indexs_valid, save_dirs, save_dirs3)
    #
    # scipy.io.savemat('./info.mat', {'pixel_freqs': freqs, 'img_labels': labels, 'indexs': indexs, 'test_indexs': test_indexs,
    #                                 'train_indexs': train_indexs, 'new_train_indexs': new_indexs_train, 'new_valid_indexs': new_indexs_valid})

    filenames = os.listdir('F:/EMWLAB/Img2Img/PlaneSAR/terrasar_SAR/sample_plane/opt')
    filenames.sort(key=lambda x: int(x[:-4]))
    train_indexs, test_indexs = split_train_test_dataset(filenames)
    aug_samples(train_indexs, ['F:/EMWLAB/Img2Img/PlaneSAR/terrasar_SAR/sample_plane/opt',
                               'F:/EMWLAB/Img2Img/PlaneSAR/terrasar_SAR/sample_plane/sar'],
                ['F:/EMWLAB/Img2Img/PlaneSAR/terrasar_SAR/sample_plane/dataset/train/opt',
                 'F:/EMWLAB/Img2Img/PlaneSAR/terrasar_SAR/sample_plane/dataset/train/sar'], 4)
    aug_samples(test_indexs, ['F:/EMWLAB/Img2Img/PlaneSAR/terrasar_SAR/sample_plane/opt',
                               'F:/EMWLAB/Img2Img/PlaneSAR/terrasar_SAR/sample_plane/sar'],
                ['F:/EMWLAB/Img2Img/PlaneSAR/terrasar_SAR/sample_plane/dataset/test/opt',
                 'F:/EMWLAB/Img2Img/PlaneSAR/terrasar_SAR/sample_plane/dataset/test/sar'], 0)
