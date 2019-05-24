# ---------------------------------------------------------
# Tensorflow U-Net Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import cv2
import logging
import tifffile as tiff

from utils import imshow

logger = logging.getLogger(__name__)  # logger
logger.setLevel(logging.INFO)


class Dataset(object):
    def __init__(self, name='EMSegmentation', log_dir=None):
        self.name = name
        self.dataset_path = '../../Data/EMSegmentation'

        self.train_imgs = tiff.imread(os.path.join(self.dataset_path, 'train-volume.tif'))
        self.train_labels = tiff.imread(os.path.join(self.dataset_path, 'train-labels.tif'))
        self.test_imgs = tiff.imread(os.path.join(self.dataset_path, 'test-volume.tif'))

        self.num_train = self.train_imgs.shape[0]
        self.num_test = self.test_imgs.shape[0]

        self.init_logger(log_dir)

    def init_logger(self, log_dir):
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

        # file handler
        file_handler = logging.FileHandler(os.path.join(log_dir, 'dataset.log'))
        file_handler.setFormatter(formatter)

        # stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        # add handlers
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    def info(self, show_img=False, use_logging=True):
        if use_logging:
            logger.info('- Training-img set:\t{}'.format(self.train_imgs.shape))
            logger.info('- Training-label set:\t{}'.format(self.train_labels.shape))
            logger.info('- Test-img set:\t\t{}'.format(self.test_imgs.shape))

            logger.info('- image shape:\t\t{}'.format(self.train_imgs[0].shape))
        else:
            print('- Training-img set:\t{}'.format(self.train_imgs.shape))
            print('- Training-label set:\t{}'.format(self.train_labels.shape))
            print('- Test-img set:\t\t{}'.format(self.test_imgs.shape))
            print('- image shape:\t\t{}'.format(self.train_imgs[0].shape))

        if show_img:
            for i in range(data.num_train):
                img, label = data.train_imgs[i], data.train_labels[i]
                imshow(img, label)

if __name__ == '__main__':
    data = Dataset()

    for i in range(data.num_train):
        img, label = data.train_imgs[i], data.train_labels[i]
        imshow(img, label)

