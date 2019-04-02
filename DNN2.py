from age.base_nn import dense, dense_bn_activation, dropout
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from age.preprocess import y2cate
from sklearn.metrics import confusion_matrix


class DNN:
    def __init__(self, X_train, y_train, X_val, y_val, batch_size=32, epoch=100, lr=1e-3):
        # __init__()中的变量
        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = lr
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        # 对__init__()中的变量进行变换
        self.n, self.input_dim = self.X_train.shape
        self.batch_num = self.n // self.batch_size + 1
        self.output_dim = self.y_train.shape[1]

        # 下面的code，需要用到的变量
        self.x_, self.y_ = None, None
        self.loss, self.op = None, None
        self.current_index = 0
        self.batch_x, self.batch_y = None, None
        self.cate = None
        self.merged, self.learning_rate = None, None
        self.keep_prob, self.outputs_cate = None, None
        self.acc = None

    def build(self):
        # placeholder
        self.x_ = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='x')
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.output_dim], name='y')
        self.learning_rate = tf.placeholder(tf.float32, name='lr')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # 网络结构(一)
        dropout_layer1 = dropout(self.x_, keep_prob=self.keep_prob, name='dropout_layer1')
        layer1_1 = dense_bn_activation(dropout_layer1, 32, tf.nn.tanh, name='layer1_1')
        dropout_layer1_1 = dropout(layer1_1, keep_prob=self.keep_prob)
        layer1_2 = dense_bn_activation(dropout_layer1_1, 8, tf.nn.tanh, name='layer1_2')
        dropout_layer1_2 = dropout(layer1_2, keep_prob=self.keep_prob)
        # layer1_3 = dense_bn_activation(dropout_layer1_2, 8, tf.nn.tanh, name='layer1_3')
        # dropout_layer1_3 = dropout(layer1_3, keep_prob=self.keep_prob)
        logits = dense(dropout_layer1_2, self.output_dim, activation=None, name='output_category')
        self.outputs_cate = tf.nn.softmax(logits)

        with tf.name_scope('optimizer'):
            # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss_y = tf.reduce_mean(tf.constant([1, 1, 5], tf.float32, shape=[1, self.output_dim])
                                        * self.y_ * (-tf.log(self.outputs_cate)))
            # loss_y = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=logits))
            # + 5 * sum(reg_losses)
            self.loss = loss_y
            self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_, axis=1),
                                                       tf.argmax(self.outputs_cate, axis=1)), tf.float32))
            self.op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.name_scope('summary'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.acc)
            self.merged = tf.summary.merge_all()

    def reset(self):
        self.current_index = 0
        index = list(range(self.n))
        np.random.shuffle(index)
        self.X_train = self.X_train[index]
        self.y_train = self.y_train[index]

    def next_batch(self):
        next_index = self.current_index + self.batch_size
        if next_index <= self.n:
            self.batch_x = self.X_train[self.current_index:next_index]
            self.batch_y = self.y_train[self.current_index:next_index]
        else:
            self.batch_x = self.X_train[self.current_index:]
            self.batch_y = self.y_train[self.current_index:]
        self.current_index = next_index

    def train(self, sess):
        train_writer = tf.summary.FileWriter('C:/Users/tianping/Desktop/Age/train/', sess.graph)
        val_writer = tf.summary.FileWriter('C:/Users/tianping/Desktop/Age/val/', sess.graph)
        sess.run(tf.global_variables_initializer())
        count = 0
        for e in tqdm(range(self.epoch), 'epoch'):
            self.reset()
            for _ in range(self.batch_num):
                self.next_batch()

                feed_dict = {self.x_: self.batch_x, self.y_: self.batch_y,
                             self.learning_rate: self.lr, self.keep_prob: 0.5}
                _, train_loss = sess.run([self.op, self.merged], feed_dict=feed_dict)

                feed_dict = {self.x_: self.X_val, self.y_: self.y_val,
                             self.keep_prob: 1.0}
                val_loss = sess.run(self.merged, feed_dict=feed_dict)
                count += 1
                train_writer.add_summary(train_loss, count)
                val_writer.add_summary(val_loss, count)

            if e % 150 == 0:
                self.lr *= 0.1

    def evaluate(self, sess, X_test, y_test):
        feed_dict = {self.x_: X_test, self.y_: y_test,
                     self.keep_prob: 1.0}
        cate_pre = sess.run(self.outputs_cate, feed_dict=feed_dict)
        matrix = confusion_matrix(y_true=np.argmax(y_test, axis=1),
                                  y_pred=np.argmax(cate_pre, axis=1))
        print(matrix)
        ACC = np.mean(np.argmax(cate_pre, axis=1) == np.argmax(y_test, axis=1))
        return ACC

    def evaluate2(self, sess, X_test):
        feed_dict = {self.x_: X_test,
                     self.keep_prob: 1.0}
        cate_pre = sess.run(self.outputs_cate, feed_dict=feed_dict)
        return cate_pre

    def save(self):
        pass

    def load(self):
        pass
