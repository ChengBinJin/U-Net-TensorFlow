# ---------------------------------------------------------
# Tensorflow U-Net Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime

from dataset import Dataset
from model import Model
from solver import Solver
from utils import make_folders

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_string('dataset', 'EMSegmentation', 'dataset name, default: EMSegmentation')
tf.flags.DEFINE_integer('batch_size', 4, 'batch size for one iteration, default: 4')
tf.flags.DEFINE_bool('is_train', True, 'training or inference mode, default: True')
tf.flags.DEFINE_float('learning_rate', 1e-3, 'initial learning rate for optimizer, default: 0.001')
tf.flags.DEFINE_float('weight_decay', 1e-5, 'weight decay for model to handle overfitting, default: 0.00001')
tf.flags.DEFINE_integer('iters', 200, 'number of iterations for one epoch, default: 20,000')
tf.flags.DEFINE_integer('print_freq', 10, 'print frequency for loss information, default: 10')
tf.flags.DEFINE_integer('eval_freq', 20, 'evaluation frequency for batch accuracy, default: 200')
tf.flags.DEFINE_string('load_model', None, 'folder of saved model that you wish to continue training '
                                           '(e.g. 20190524-1606), default: None')

logger = logging.getLogger(__name__)  # logger
logger.setLevel(logging.INFO)

def init_logger(log_dir, is_train=True):
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

    # file handler
    file_handler = logging.FileHandler(os.path.join(log_dir, 'main.log'))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # add handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    if is_train:
        logger.info('gpu_index: \t\t{}'.format(FLAGS.gpu_index))
        logger.info('dataset: \t\t{}'.format(FLAGS.dataset))
        logger.info('batch_size: \t\t{}'.format(FLAGS.batch_size))
        logger.info('is_train: \t\t{}'.format(FLAGS.is_train))
        logger.info('learning_rate: \t{}'.format(FLAGS.learning_rate))
        logger.info('weight_decay: \t\t{}'.format(FLAGS.weight_decay))
        logger.info('iters: \t\t{}'.format(FLAGS.iters))
        logger.info('print_freq: \t\t{}'.format(FLAGS.print_freq))
        logger.info('eval_freq: \t\t{}'.format(FLAGS.eval_freq))
        logger.info('load_model: \t\t{}'.format(FLAGS.load_model))
    else:
        print('-- gpu_index: \t\t{}'.format(FLAGS.gpu_index))
        print('-- dataset: \t\t{}'.format(FLAGS.dataset))
        print('-- batch_size: \t\t{}'.format(FLAGS.batch_size))
        print('-- is_train: \t\t{}'.format(FLAGS.is_train))
        print('-- learning_rate: \t{}'.format(FLAGS.learning_rate))
        print('-- weight_decay: \t\t{}'.format(FLAGS.weight_decay))
        print('-- iters: \t\t{}'.format(FLAGS.iters))
        print('-- print_freq: \t\t{}'.format(FLAGS.print_freq))
        print('-- eval_freq: \t\t{}'.format(FLAGS.eval_freq))
        print('-- load_model: \t\t{}'.format(FLAGS.load_model))


def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_index

    # Initialize model and log folders
    if FLAGS.load_model is None:
        cur_time = datetime.now().strftime("%Y%m%d-%H%M")
    else:
        cur_time = FLAGS.load_model

    model_dir, log_dir = make_folders(is_train=FLAGS.is_train, cur_time=cur_time)
    init_logger(log_dir=log_dir, is_train=FLAGS.is_train)

    data = Dataset(name=FLAGS.dataset, log_dir=log_dir)
    data.info(use_logging=True, log_dir=log_dir)

    sess = tf.Session()  # Initialize session
    model = Model(input_shape=data.input_shape,
                  output_shape=data.output_shape,
                  lr=FLAGS.learning_rate,
                  weight_decay=FLAGS.weight_decay,
                  total_iters=FLAGS.iters,
                  is_train=FLAGS.is_train,
                  log_dir=log_dir,
                  name='U-Net')
    solver = Solver(sess, model)
    solver.init()

    if FLAGS.is_train:
        train(data, solver)


def train(data, solver):
    for iter_time in range(FLAGS.iters):
        x_batch, y_batch = data.random_batch(batch_size=FLAGS.batch_size, idx=iter_time)
        _, total_loss, data_loss, reg_term, summary = solver.train(x_batch, y_batch)

        if np.mod(iter_time, FLAGS.print_freq) == 0:
            msg = '{}/{}: \tTotal loss: {:.2f}, \tData loss: {:.2f}, \tReg. term: {:.2f}'
            print(msg.format(iter_time, FLAGS.iters, total_loss, data_loss, reg_term))

        if np.mod(iter_time, FLAGS.eval_freq) == 0:
            x_batch, y_batch = data.random_batch(batch_size=FLAGS.batch_size,
                                                 idx=np.random.randint(low=0, high=FLAGS.iters))
            acc, total_loss, data_loss, reg_term = solver.test(x_batch, y_batch)

            msg = 'Accuracy: {:.2f}%, \tTotal loss: {:.2f}, \tData loss: {:.2f}, \tReg. term: {:.2f}'
            print(msg.format(acc, total_loss, data_loss, reg_term))


def test():
    print('Hello test!')

if __name__ == '__main__':
    tf.app.run()
