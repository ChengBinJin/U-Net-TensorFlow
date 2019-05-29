import os
import cv2
import numpy as np
import tensorflow as tf


class Solver(object):
    def __init__(self, sess, model, mean_value):
        self.sess = sess
        self.model = model
        self.mean_value = mean_value

    def train(self, x, y):
        feed = {
            self.model.inp_img: np.expand_dims(x, axis=3),
            self.model.out_img: y,
            self.model.keep_prob: 0.5
        }

        train_op = self.model.train_op
        total_loss = self.model.total_loss
        data_loss = self.model.data_loss
        reg_term = self.model.reg_term
        pred_cls = self.model.pred_cls
        summary = self.model.summary_op

        return self.sess.run([train_op, total_loss, data_loss, reg_term, summary, pred_cls], feed_dict=feed)

    def test(self, x, y, batch_size=4, is_train=True):
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

        if is_train:
            summary = self.sess.run(self.model.val_acc_op, feed_dict={self.model.val_acc: avg_acc})
            return avg_acc, summary
        else:
            return avg_acc

    def save_imgs(self, x_imgs, pred_imgs, y_imgs, iter_time, sample_dir=None, margin=5):
        num_cols = 3
        _, H1, W1 = x_imgs.shape
        N, H2, W2 = pred_imgs.shape
        margin = int(0.5 * (H1 - H2))

        canvas = np.zeros((N * H1, 1 * W1 + (num_cols - 1) * W2), dtype=np.uint8)
        x_imgs += self.mean_value
        y_imgs *= 255
        pred_imgs *= 255

        print('y_imgs.shape: {}'.format(y_imgs.shape))

        print('N: {}'.format(N))
        print('H1: {}, W1: {}'.format(H1, W1))
        print('H2: {}, W2: {}'.format(H2, W2))

        for idx in range(N):
            canvas[idx*H1:(idx+1)*H1, 0:W1] = x_imgs[idx]
            canvas[idx*H1+margin:idx*H1+margin+H2, W1:W1+W2] = pred_imgs[idx]
            canvas[idx*H1+margin:idx*H1+margin+H2, W1+W2:] = y_imgs[idx]

        cv2.imwrite(os.path.join(sample_dir, str(iter_time).zfill(5) + '.png'), canvas)


    def init(self):
        self.sess.run(tf.global_variables_initializer())
