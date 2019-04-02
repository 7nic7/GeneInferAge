import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import roc_curve,auc
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


class network:
    def __init__(self, batch_size=32, epoch=50, lr=1e-4,
                 max_lr=0.02, enlarge_lr=1.005, reduce_lr=0.98):

        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = lr
        self.max_lr = max_lr
        self.enlarge_lr = enlarge_lr
        self.reduce_lr = reduce_lr
        self.e = 0

    def build(self, a2, a3, a4):

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 50], name='x_layer1')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 50], name='y_layer5')
        self.learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

        with tf.variable_scope('inputs_1'):

            layer2 = tf.layers.dense(self.x, a2, activation=tf.nn.tanh, name='layer2')

        with tf.variable_scope('encoder'):

            layer3 = tf.layers.dense(layer2, a3, name='layer3')  # can't change
            encoder = step_wise(layer3, n=a3)         # the most important

        with tf.variable_scope('outputs_5'):

            layer4 = tf.layers.dense(encoder, a4, activation=tf.nn.tanh, name='layer4')
            self.outputs = tf.layers.dense(layer4, 50, activation=tf.nn.sigmoid, name='layer5')

        with tf.name_scope('optimize'):

            self.loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.y,
                                                                    predictions=self.outputs))
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        tf.summary.scalar('loss', self.loss)
        self.merged = tf.summary.merge_all()

    def reset(self):

        index = np.arange(0, self.train_X.shape[0], 1)
        np.random.shuffle(index)
        self.train_X = self.train_X[index, :]
        # self.train_y = self.train_y[index]
        self.current_index = 0

    def next_batch(self):

        assert self.current_index < self.train_X.shape[0]
        batch_x = self.train_X[self.current_index:(self.current_index + self.batch_size), :]
        batch_y = batch_x

        return batch_x, batch_y

    def train(self, X, y=None, sess=None):

        self.train_X = X
        num_batch = self.train_X.shape[0] // self.batch_size
        sess.run(tf.global_variables_initializer())
        self.writer_train = tf.summary.FileWriter('G:/python_file/wine/train/', sess.graph)
        num = 0
        for _ in tqdm(range(self.epoch), desc='epoch'):
        # for _ in range(self.epoch):
            self.reset()
            for _ in range(num_batch):

                num += 1
                (batch_x, batch_y) = self.next_batch()

                feed_dict = {self.x: batch_x, self.y: batch_y,
                             self.learning_rate: self.lr}
                _, e, train_result = sess.run([self.train_op, self.loss, self.merged],
                                              feed_dict=feed_dict)
                self.writer_train.add_summary(train_result, num)
                if e > 1.01 * self.e:
                    self.lr *= self.reduce_lr
                elif e < self.e and self.lr < self.max_lr:
                    self.lr *= self.enlarge_lr
                else:
                    self.lr = self.lr
                self.e = e


def step_wise(theta, a=100, n=4):   # theta : tensor with shape of 32,
    out = tf.zeros_like(theta)
    for i in range(1, n):
        out = tf.add(tf.nn.tanh(a*(theta-tf.constant(i/n))), out)
    out = 1/2 + 1/4 * out

    return out

