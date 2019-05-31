# ---------------------------------------------------------
# Tensorflow U-Net Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import tensorflow as tf
import utils as utils
import tensorflow_utils as tf_utils

class Model(object):
    def __init__(self, input_shape, output_shape, lr=0.001, weight_decay=1e-4, total_iters=2e4, is_train=True,
                 log_dir=None, name='U-Net'):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.conv_dims = [64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024,
                          512, 512, 512, 256, 256, 256, 128, 128, 128, 64, 64, 64, 2]
        self.lr = lr
        self.weight_decay = weight_decay
        self.total_steps = total_iters
        self.start_decay_step = int(self.total_steps * 0.5)
        self.decay_steps = self.total_steps - self.start_decay_step
        self.is_train = is_train
        self.log_dir = log_dir
        self.name = name
        self.tb_lr, self.pred = None, None
        self.logger, self.file_handler, self.stream_handler = utils.init_logger(log_dir=self.log_dir,
                                                                                name=self.name,
                                                                                is_train=self.is_train)

        with tf.variable_scope(self.name):
            self._build_net()

        self._tensorboard()

    def _build_net(self):
        # Input placeholders
        self.inp_img = tf.placeholder(dtype=tf.float32, shape=[None, *self.input_shape], name='input_img')
        self.out_img = tf.placeholder(dtype=tf.uint8, shape=[None, *self.output_shape], name='output_img')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        self.val_acc = tf.placeholder(dtype=tf.float32, name='val_acc')

        # One-hot representation
        self.out_img_one_hot = tf.one_hot(indices=self.out_img, depth=2, axis=-1, dtype=tf.float32, name='one_hot')

        # U-Net building
        self.u_net()
        self.pred_cls = tf.math.argmax(input=self.pred, axis=-1)

        # Loss = data loss + regularization term
        self.data_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.out_img_one_hot, logits=self.pred))
        self.reg_term = self.weight_decay * tf.reduce_sum(
            [tf.nn.l2_loss(weight) for weight in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
        self.total_loss = self.data_loss + self.reg_term

        # Optimizer
        self.train_op = self.optimizer_fn(loss=self.total_loss, name='Adam')

        # Accuracy
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.math.equal(x=tf.cast(tf.math.argmax(self.out_img_one_hot, axis=-1), dtype=tf.uint8),
                                  y=tf.cast(self.pred_cls, dtype=tf.uint8)), dtype=tf.float32)) * 100.

    def u_net(self):
        # Stage 1
        tf_utils.print_activations(self.inp_img, logger=self.logger)
        s1_conv1 = tf_utils.conv2d(x=self.inp_img, output_dim=self.conv_dims[0], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding='VALID', initializer='He', name='s1_conv1', logger=self.logger)
        s1_conv1 = tf_utils.relu(s1_conv1, name='relu_s1_conv1', logger=self.logger)
        s1_conv2 = tf_utils.conv2d(x=s1_conv1, output_dim=self.conv_dims[1], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding='VALID', initializer='He', name='s1_conv2', logger=self.logger)
        s1_conv2 = tf_utils.relu(s1_conv2, name='relu_s1_conv2', logger=self.logger)

        # Stage 2
        s2_maxpool = tf_utils.max_pool(x=s1_conv2, name='s2_maxpool', logger=self.logger)
        s2_conv1 = tf_utils.conv2d(x=s2_maxpool, output_dim=self.conv_dims[2], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding='VALID', initializer='He', name='s2_conv1', logger=self.logger)
        s2_conv1 = tf_utils.relu(s2_conv1, name='relu_s2_conv1', logger=self.logger)
        s2_conv2 = tf_utils.conv2d(x=s2_conv1, output_dim=self.conv_dims[3], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding='VALID', initializer='He', name='s2_conv2', logger=self.logger)
        s2_conv2 = tf_utils.relu(s2_conv2, name='relu_s2_conv2', logger=self.logger)

        # Stage 3
        s3_maxpool = tf_utils.max_pool(x=s2_conv2, name='s3_maxpool', logger=self.logger)
        s3_conv1 = tf_utils.conv2d(x=s3_maxpool, output_dim=self.conv_dims[4], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding='VALID', initializer='He', name='s3_conv1', logger=self.logger)
        s3_conv1 = tf_utils.relu(s3_conv1, name='relu_s3_conv1', logger=self.logger)
        s3_conv2 = tf_utils.conv2d(x=s3_conv1, output_dim=self.conv_dims[5], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding='VALID', initializer='He', name='s3_conv2', logger=self.logger)
        s3_conv2 = tf_utils.relu(s3_conv2, name='relu_s3_conv2', logger=self.logger)

        # Stage 4
        s4_maxpool = tf_utils.max_pool(x=s3_conv2, name='s4_maxpool', logger=self.logger)
        s4_conv1 = tf_utils.conv2d(x=s4_maxpool, output_dim=self.conv_dims[6], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding='VALID', initializer='He', name='s4_conv1', logger=self.logger)
        s4_conv1 = tf_utils.relu(s4_conv1, name='relu_s4_conv1', logger=self.logger)
        s4_conv2 = tf_utils.conv2d(x=s4_conv1, output_dim=self.conv_dims[7], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding='VALID', initializer='He', name='s4_conv2', logger=self.logger)
        s4_conv2 = tf_utils.relu(s4_conv2, name='relu_s4_conv2', logger=self.logger)
        s4_conv2_drop = tf_utils.dropout(x=s4_conv2, keep_prob=self.keep_prob, name='s4_conv2_dropout',
                                         logger=self.logger)

        # Stage 5
        s5_maxpool = tf_utils.max_pool(x=s4_conv2, name='s5_maxpool', logger=self.logger)
        s5_conv1 = tf_utils.conv2d(x=s5_maxpool, output_dim=self.conv_dims[8], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding='VALID', initializer='He', name='s5_conv1', logger=self.logger)
        s5_conv1 = tf_utils.relu(s5_conv1, name='relu_s5_conv1', logger=self.logger)
        s5_conv2 = tf_utils.conv2d(x=s5_conv1, output_dim=self.conv_dims[9], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding='VALID', initializer='He', name='s5_conv2', logger=self.logger)
        s5_conv2 = tf_utils.relu(s5_conv2, name='relu_s5_conv2', logger=self.logger)
        s5_conv2_drop = tf_utils.dropout(x=s5_conv2, keep_prob=self.keep_prob, name='s5_conv2_dropout',
                                         logger=self.logger)

        # Stage 6
        s6_deconv1 = tf_utils.deconv2d(x=s5_conv2_drop, output_dim=self.conv_dims[10], k_h=2, k_w=2, initializer='He',
                                       name='s6_deconv1')
        # Cropping
        h1, w1 = s4_conv2_drop.get_shape().as_list()[1:3]
        h2, w2 = s6_deconv1.get_shape().as_list()[1:3]
        s4_conv2_crop = tf.image.crop_to_bounding_box(image=s4_conv2_drop,
                                                      offset_height=int(0.5 * (h1 - h2)),
                                                      offset_width=int(0.5 * (w1 - w2)),
                                                      target_height=h2,
                                                      target_width=w2)
        tf_utils.print_activations(s4_conv2_crop, logger=self.logger)

        s6_concat = tf_utils.concat(values=[s4_conv2_crop, s6_deconv1], axis=3, name='s6_concat', logger=self.logger)
        s6_conv2 = tf_utils.conv2d(x=s6_concat, output_dim=self.conv_dims[11], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding='VALID', initializer='He', name='s6_conv2', logger=self.logger)
        s6_conv2 = tf_utils.relu(s6_conv2, name='relu_s6_conv2', logger=self.logger)
        s6_conv3 = tf_utils.conv2d(x=s6_conv2, output_dim=self.conv_dims[12], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding='VALID', initializer='He', name='s6_conv3', logger=self.logger)
        s6_conv3 = tf_utils.relu(s6_conv3, name='relu_s6_conv3', logger=self.logger)

        # Stage 7
        s7_deconv1 = tf_utils.deconv2d(x=s6_conv3, output_dim=self.conv_dims[13], k_h=2, k_w=2, initializer='He',
                                       name='s7_deconv1')
        s7_deconv1 = tf_utils.relu(s7_deconv1, name='relu_s7_deconv1', logger=self.logger)
        # Cropping
        h1, w1 = s3_conv2.get_shape().as_list()[1:3]
        h2, w2 = s7_deconv1.get_shape().as_list()[1:3]
        s3_conv2_crop = tf.image.crop_to_bounding_box(image=s3_conv2,
                                                      offset_height=int(0.5 * (h1 - h2)),
                                                      offset_width=int(0.5 * (w1 - w2)),
                                                      target_height=h2,
                                                      target_width=w2)
        tf_utils.print_activations(s3_conv2_crop, logger=self.logger)

        s7_concat = tf_utils.concat(values=[s3_conv2_crop, s7_deconv1], axis=3, name='s7_concat', logger=self.logger)
        s7_conv2 = tf_utils.conv2d(x=s7_concat, output_dim=self.conv_dims[14], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding='VALID', initializer='He', name='s7_conv2', logger=self.logger)
        s7_conv2 = tf_utils.relu(s7_conv2, name='relu_s7_conv2', logger=self.logger)
        s7_conv3 = tf_utils.conv2d(x=s7_conv2, output_dim=self.conv_dims[15], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding='VALID', initializer='He', name='s7_conv3', logger=self.logger)
        s7_conv3 = tf_utils.relu(s7_conv3, name='relu_s7_conv3', logger=self.logger)

        # Stage 8
        s8_deconv1 = tf_utils.deconv2d(x=s7_conv3, output_dim=self.conv_dims[16], k_h=2, k_w=2, initializer='He',
                                       name='s8_deconv1')
        s8_deconv1 = tf_utils.relu(s8_deconv1, name='relu_s8_deconv1', logger=self.logger)
        # Cropping
        h1, w1 = s2_conv2.get_shape().as_list()[1:3]
        h2, w2 = s8_deconv1.get_shape().as_list()[1:3]
        s2_conv2_crop = tf.image.crop_to_bounding_box(image=s2_conv2,
                                                      offset_height=int(0.5 * (h1 - h2)),
                                                      offset_width=int(0.5 * (w1 - w2)),
                                                      target_height=h2,
                                                      target_width=w2)
        tf_utils.print_activations(s2_conv2_crop, logger=self.logger)

        s8_concat = tf_utils.concat(values=[s2_conv2_crop, s8_deconv1], axis=3, name='s8_concat', logger=self.logger)
        s8_conv2 = tf_utils.conv2d(x=s8_concat, output_dim=self.conv_dims[17], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding='VALID', initializer='He', name='s8_conv2', logger=self.logger)
        s8_conv2 = tf_utils.relu(s8_conv2, name='relu_s8_conv2', logger=self.logger)
        s8_conv3 = tf_utils.conv2d(x=s8_conv2, output_dim=self.conv_dims[18], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding='VALID', initializer='He', name='s8_conv3', logger=self.logger)
        s8_conv3 = tf_utils.relu(s8_conv3, name='relu_conv3', logger=self.logger)

        # Stage 9
        s9_deconv1 = tf_utils.deconv2d(x=s8_conv3, output_dim=self.conv_dims[19], k_h=2, k_w=2, initializer='He',
                                       name='s9_deconv1')
        s9_deconv1 = tf_utils.relu(s9_deconv1, name='relu_s9_deconv1', logger=self.logger)
        # Cropping
        h1, w1 = s1_conv2.get_shape().as_list()[1:3]
        h2, w2 = s9_deconv1.get_shape().as_list()[1:3]
        s1_conv2_crop = tf.image.crop_to_bounding_box(image=s1_conv2,
                                                      offset_height=int(0.5 * (h1 - h2)),
                                                      offset_width=int(0.5 * (w1 - w2)),
                                                      target_height=h2,
                                                      target_width=w2)
        tf_utils.print_activations(s1_conv2_crop, logger=self.logger)

        s9_concat = tf_utils.concat(values=[s1_conv2_crop, s9_deconv1], axis=3, name='s9_concat', logger=self.logger)
        s9_conv2 = tf_utils.conv2d(x=s9_concat, output_dim=self.conv_dims[20], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding='VALID', initializer='He', name='s9_conv2', logger=self.logger)
        s9_conv2 = tf_utils.relu(s9_conv2, name='relu_s9_conv2', logger=self.logger)
        s9_conv3 = tf_utils.conv2d(x=s9_conv2, output_dim=self.conv_dims[21], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding='VALID', initializer='He', name='s9_conv3', logger=self.logger)
        s9_conv3 = tf_utils.relu(s9_conv3, name='relu_s9_conv3', logger=self.logger)
        self.pred = tf_utils.conv2d(x=s9_conv3, output_dim=self.conv_dims[22], k_h=1, k_w=1, d_h=1, d_w=1,
                                    padding='SAME', initializer='He', name='output', logger=self.logger)

    def optimizer_fn(self,loss, name=None):
        with tf.variable_scope(name):
            global_step = tf.Variable(0, dtype=tf.float32, trainable=False)
            start_learning_rate = self.lr
            end_learning_rate = 0.
            start_decay_stp = self.start_decay_step
            decay_steps = self.decay_steps

            learning_rate = (tf.where(tf.greater_equal(global_step, start_decay_stp),
                                      tf.train.polynomial_decay(start_learning_rate,
                                                                global_step - start_decay_stp,
                                                                decay_steps, end_learning_rate, power=1.0),
                                      start_learning_rate))
            self.tb_lr = tf.summary.scalar('learning_rate', learning_rate)

            learn_step = tf.train.AdamOptimizer(learning_rate, beta1=0.99).minimize(loss, global_step=global_step)

        return learn_step

    def _tensorboard(self):
        self.tb_total = tf.summary.scalar('Loss/total', self.total_loss)
        self.tb_data = tf.summary.scalar('Loss/data', self.data_loss)
        self.tb_reg = tf.summary.scalar('Loss/reg_term', self.reg_term)
        self.summary_op = tf.summary.merge(inputs=[self.tb_lr, self.tb_total, self.tb_data, self.tb_reg])

        self.val_acc_op = tf.summary.scalar('Acc', self.val_acc)

    def release_handles(self):
        utils.release_handles(self.logger, self.file_handler, self.stream_handler)
