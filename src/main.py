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
tf.flags.DEFINE_integer('iters', 100, 'number of iterations for one epoch, default: 20,000')
tf.flags.DEFINE_integer('print_freq', 10, 'print frequency for loss information, default: 10')
tf.flags.DEFINE_integer('sample_freq', 10, 'sample frequence for checking qualitative evaluation, default: 100')
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
        logger.info('sample_freq: \t\t{}'.format(FLAGS.sample_freq))
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
        print('-- sample_freq: \t\t{}'.format(FLAGS.sample_freq))
        print('-- eval_freq: \t\t{}'.format(FLAGS.eval_freq))
        print('-- load_model: \t\t{}'.format(FLAGS.load_model))


def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_index

    # Initialize model and log folders
    if FLAGS.load_model is None:
        cur_time = datetime.now().strftime("%Y%m%d-%H%M")
    else:
        cur_time = FLAGS.load_model

    model_dir, log_dir, sample_dir = make_folders(is_train=FLAGS.is_train, cur_time=cur_time)
    init_logger(log_dir=log_dir, is_train=FLAGS.is_train)

    # Initilize dataset
    data = Dataset(name=FLAGS.dataset, log_dir=log_dir)
    data.info(use_logging=True, log_dir=log_dir)

    # Initialize session
    sess = tf.Session()

    # Initilize model
    model = Model(input_shape=data.input_shape,
                  output_shape=data.output_shape,
                  lr=FLAGS.learning_rate,
                  weight_decay=FLAGS.weight_decay,
                  total_iters=FLAGS.iters,
                  is_train=FLAGS.is_train,
                  log_dir=log_dir,
                  name='U-Net')

    # Initilize solver
    solver = Solver(sess, model, data.mean_value)
    saver = tf.train.Saver(max_to_keep=1)

    if FLAGS.is_train:
        train(data, solver, saver, model_dir, log_dir, sample_dir)


def train(data, solver, saver, model_dir, log_dir, sample_dir):
    best_acc = 0.
    num_evals = 0
    tb_writer = tf.summary.FileWriter(log_dir, graph_def=solver.sess.graph_def)
    solver.init()

    for iter_time in range(FLAGS.iters):
        x_batch, y_batch = data.random_batch(batch_size=FLAGS.batch_size, idx=iter_time)
        _, total_loss, data_loss, reg_term, summary, pred_cls = solver.train(x_batch, y_batch)

        # Write to tensorbard
        tb_writer.add_summary(summary, iter_time)
        tb_writer.flush()

        if np.mod(iter_time, FLAGS.print_freq) == 0:
            msg = '{}/{}: \tTotal loss: {:.3f}, \tData loss: {:.3f}, \tReg. term: {:.3f}'
            print(msg.format(iter_time, FLAGS.iters, total_loss, data_loss, reg_term))

        if np.mod(iter_time, FLAGS.sample_freq) == 0:
            solver.save_imgs(x_batch, pred_cls, y_batch, iter_time, sample_dir)

        if np.mod(iter_time, FLAGS.eval_freq) == 0:
            x_batch, y_batch = data.random_batch(batch_size=FLAGS.batch_size * 20,
                                                 idx=np.random.randint(low=0, high=FLAGS.iters))
            acc, summary = solver.test(x_batch, y_batch, batch_size=FLAGS.batch_size, is_train=True)
            print('Evaluation! \tAcc: {:.3f}'.format(acc))

            # Write to tensorboard
            tb_writer.add_summary(summary, num_evals)
            tb_writer.flush()
            num_evals += 1

            if acc > best_acc:
                logger.info('Acc: {:.3f}, Best Acc: {:.3f}'.format(acc, best_acc))
                save_model(saver, solver, model_dir, iter_time)
                best_acc = acc


def test():
    print('Hello test!')


def save_model(saver, solver, model_dir, iter_time):
    saver.save(solver.sess, os.path.join(model_dir, 'model'), global_step=iter_time)
    logger.info(' [*] Model saved! Iter: {}'.format(iter_time))


if __name__ == '__main__':
    tf.app.run()
