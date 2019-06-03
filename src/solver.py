import os
import cv2
import numpy as np
import tensorflow as tf

import utils as utils


class Solver(object):
    def __init__(self, sess, model, mean_value):
        self.sess = sess
        self.model = model
        self.mean_value = mean_value

    def train(self, x, y, wmap):
        feed = {
            self.model.inp_img: np.expand_dims(x, axis=3),
            self.model.out_img: y,
            self.model.weight_map: wmap,
            self.model.keep_prob: 0.5
        }

        train_op = self.model.train_op
        total_loss = self.model.total_loss
        avg_data_loass = self.model.avg_data_loss
        weighted_data_loss = self.model.weighted_data_loss
        reg_term = self.model.reg_term
        pred_cls = self.model.pred_cls
        summary = self.model.summary_op

        return self.sess.run([train_op, total_loss, avg_data_loass, weighted_data_loss, reg_term, summary, pred_cls],
                             feed_dict=feed)

    def evalate(self, x, y, batch_size=4):
        print(' [*] Evaluation...')

        num_test = x.shape[0]
        avg_acc = 0.

        for i_start in range(0, num_test, batch_size):
            if i_start + batch_size < num_test:
                i_end = i_start + batch_size
            else:
                i_end = num_test - 1

            x_batch = x[i_start:i_end]
            y_batch = y[i_start:i_end]

            feed = {
                self.model.inp_img: np.expand_dims(x_batch, axis=3),
                self.model.out_img: y_batch,
                self.model.keep_prob: 1.0
            }

            acc_op = self.model.accuracy
            avg_acc += self.sess.run(acc_op, feed_dict=feed)

        avg_acc = np.float32(avg_acc / np.ceil(num_test / batch_size))
        summary = self.sess.run(self.model.val_acc_op, feed_dict={self.model.val_acc: avg_acc})

        return avg_acc, summary

    def test(self, x, iter_time, angle, test_dir, is_save=False):
        feed = {
            self.model.inp_img: np.expand_dims(x, axis=3),
            self.model.keep_prob: 1.0
        }

        preds = self.sess.run(self.model.pred, feed_dict=feed)
        pred = utils.merge_preds(preds, idx=iter_time, angle=angle, test_dir=test_dir, is_save=is_save)

        return pred

    def save_imgs(self, x_imgs, pred_imgs, y_imgs, iter_time, sample_dir=None, border=5):
        num_cols = 3
        _, H1, W1 = x_imgs.shape
        N, H2, W2 = pred_imgs.shape
        margin = int(0.5 * (H1 - H2))

        canvas = np.zeros((N*H1+(N+1)*border, 1*W1+(num_cols-1)*W2+(num_cols+1)*border), dtype=np.uint8)
        for idx in range(N):
            canvas[(idx+1)*border+idx*H1:(idx+1)*border+(idx+1)*H1, border:border+W1] = \
                x_imgs[idx] + self.mean_value
            canvas[(idx+1)*border+idx*H1+margin:(idx+1)*border+idx*H1+margin+H2, 2*border+W1:2*border+W1+W2] = \
                pred_imgs[idx] * 255
            canvas[(idx+1)*border+idx*H1+margin:(idx+1)*border+idx*H1+margin+H2, 3*border+W1+W2:3*border+W1+2*W2] = \
                y_imgs[idx] * 255

        cv2.imwrite(os.path.join(sample_dir, str(iter_time).zfill(5) + '.png'), canvas)

    def save_acc_record(self, acc):
        self.sess.run(self.model.save_best_acc_op, feed_dict={self.model.val_acc: acc})

    def load_acc_record(self):
        best_acc = self.sess.run(self.model.best_acc_record)
        return best_acc

    def init(self):
        self.sess.run(tf.global_variables_initializer())
