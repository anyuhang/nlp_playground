import tensorflow as tf
import numpy as np
from data_helpers import get_device


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, dense0_dim=512, dense1_dim=256):
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)

        with tf.name_scope('embedding'):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size,
                                   embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        pooled_outputs = []

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                h = tf.nn.softmax(tf.nn.bias_add(conv, b), name="relu")

                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")

                ''' top-5 version
                h_squeeze = tf.squeeze(h, axis=[2])
                h_r = tf.transpose(h_squeeze,  perm=[0, 2, 1])
                values, indices = tf.nn.top_k(h_r, 5)
                pooled = tf.transpose(values, perm=[0, 2, 1])
                '''

                pooled_outputs.append(pooled)

        ''' top-5 version
        num_filters_total = num_filters * len(filter_sizes) * 5
        '''

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 1)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope("dense0"):
            w_0 = tf.Variable(tf.truncated_normal([num_filters_total, dense0_dim], stddev=0.1), name="dense0_w")
            b_0 = tf.Variable(tf.constant(0.1, shape=[dense0_dim]), name="dense0_b")
            wx_plus_b_0 = tf.matmul(self.h_drop, w_0) + b_0
            self.dense0 = tf.nn.relu(wx_plus_b_0)

        with tf.name_scope("dense1"):
            w_1 = tf.Variable(tf.truncated_normal([dense0_dim, dense1_dim], stddev=0.1), name="dense1_w")
            b_1 = tf.Variable(tf.constant(0.1, shape=[dense1_dim]), name="dense1_b")
            wx_plus_b_1 = tf.matmul(self.dense0, w_1) + b_1
            self.dense1 = tf.nn.relu(wx_plus_b_1)

        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[dense1_dim, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())

            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(self.dense1, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            self.correct_sum = tf.reduce_sum(tf.cast(correct_predictions, "int64"), name="correct_sum")