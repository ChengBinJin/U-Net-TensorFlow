import numpy as np
import tensorflow as tf


class Solver(object):
    def __init__(self, sess, model):
        self.sess = sess
        self.model = model

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
        summary = self.model.summary_op

        return self.sess.run([train_op, total_loss, data_loss, reg_term, summary], feed_dict=feed)

    def test(self, x, y):
        feed = {
            self.model.inp_img: np.expand_dims(x, axis=3),
            self.model.out_img: y,
            self.model.keep_prob: 1.0
        }

        acc = self.model.accuracy
        total_loss = self.model.total_loss
        data_loss = self.model.data_loss
        reg_term = self.model.reg_term

        return self.sess.run([acc, total_loss, data_loss, reg_term], feed_dict=feed)


    def init(self):
        self.sess.run(tf.global_variables_initializer())
