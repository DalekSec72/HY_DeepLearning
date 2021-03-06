# -*- coding: utf-8 -*-

# 2021 HYU. CSE
# Taehun Kim <th6424@gmail.com>

import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score

learning_rate = 0.001
training_epochs = 25
batch_size = 128


def get_batch_data(shuffled_idx, batch_size, data, labels, start_idx):
    idx = shuffled_idx[start_idx:start_idx + batch_size]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def build_CNN_classifier(x):
    x_image = x

    W1 = tf.get_variable(name="W1", shape=[5, 5, 3, 32], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable(name="b1", shape=[32], initializer=tf.contrib.layers.xavier_initializer())
    c1 = tf.nn.conv2d(x_image, W1, strides=[1, 1, 1, 1], padding='SAME')
    l1 = tf.nn.relu(tf.nn.bias_add(c1, b1))
    l1 = tf.nn.dropout(l1, rate=rate)
    l1_pool = tf.nn.max_pool(l1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    W2 = tf.get_variable(name="W2", shape=[5, 5, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable(name="b2", shape=[64], initializer=tf.contrib.layers.xavier_initializer())
    c2 = tf.nn.conv2d(l1_pool, W2, strides=[1, 1, 1, 1], padding='SAME')
    l2 = tf.nn.relu(tf.nn.bias_add(c2, b2))
    l2 = tf.nn.dropout(l2, rate=rate)
    l2_pool = tf.nn.max_pool(l2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    W3 = tf.get_variable(name="W3", shape=[5, 5, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable(name="b3", shape=[128], initializer=tf.contrib.layers.xavier_initializer())
    c3 = tf.nn.conv2d(l2_pool, W3, strides=[1, 1, 1, 1], padding='SAME')
    l3 = tf.nn.relu(tf.nn.bias_add(c3, b3))
    l3 = tf.nn.dropout(l3, rate=rate)
    l3_pool = tf.nn.max_pool(l3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    l3_flat = tf.reshape(l3_pool, [-1, 4 * 4 * 128])

    W_fc1 = tf.get_variable(name="W_fc1", shape=[4 * 4 * 128, 10], initializer=tf.contrib.layers.xavier_initializer())
    b_fc1 = tf.get_variable(name="b_fc1", shape=[10], initializer=tf.contrib.layers.xavier_initializer())
    logits = tf.nn.bias_add(tf.matmul(l3_flat, W_fc1), b_fc1)
    hypothesis = tf.nn.softmax(logits)

    return hypothesis, logits


ckpt_path = "output/"

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])
rate = tf.placeholder(tf.float32)

x_train = np.load("data/x_train.npy")
y_train = np.load("data/y_train.npy")

dev_num = len(x_train) // 4

x_dev = x_train[:dev_num]
y_dev = y_train[:dev_num]

x_train = x_train[dev_num:]
y_train = y_train[dev_num:]

y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_dev_one_hot = tf.squeeze(tf.one_hot(y_dev, 10), axis=1)

y_pred, logits = build_CNN_classifier(x)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_step = tf.train.AdamOptimizer(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-08,
    use_locking=False,
    name='Adam'
).minimize(cost)

total_batch = int(len(x_train) / batch_size) if len(x_train) % batch_size == 0 else int(len(x_train) / batch_size) + 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("????????????")

    for epoch in range(training_epochs):
        print("Epoch", epoch + 1)
        start = 0
        shuffled_idx = np.arange(0, len(x_train))
        np.random.shuffle(shuffled_idx)

        for i in range(total_batch):
            batch_data, batch_labels = get_batch_data(
                shuffled_idx, batch_size, x_train, y_train_one_hot.eval(), i * batch_size
            )
            sess.run(train_step, feed_dict={x: batch_data, y: batch_labels, rate: 0.25})

    saver = tf.train.Saver()
    saver.save(sess, ckpt_path)
    saver.restore(sess, ckpt_path)

    y_prediction = np.argmax(y_pred.eval(feed_dict={x: x_dev, rate: 0}), 1)
    y_true = np.argmax(y_dev_one_hot.eval(), 1)
    dev_f1 = f1_score(y_true, y_prediction, average="weighted")  # f1 ????????? ??????
    print("dev ????????? f1 score: %f" % dev_f1)

    # ????????? ???????????? ?????????
    x_test = np.load("data/x_test.npy")
    test_logits = y_pred.eval(feed_dict={x: x_test, rate: 0})
    np.save("result", test_logits)
