# -*- coding: utf-8 -*-

# 2021 HYU. CSE
# Taehun Kim <th6424@gmail.com>

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist/data/', one_hot=True)

batch_size = 100
learning_rate = 0.01
epoch_num = 20
noise_level = 0.9
n_input = 784
n_hidden1 = 256
n_hidden2 = 32

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_input])

W1_encode = tf.Variable(tf.random_uniform([n_input, n_hidden1], -1, 1))
b1_encode = tf.Variable(tf.random_uniform([n_hidden1], -1, 1))
encoder1 = tf.nn.sigmoid(tf.add(tf.matmul(X, W1_encode), b1_encode))

W2_encode = tf.Variable(tf.random_uniform([n_hidden1, n_hidden2], -1, 1))
b2_encode = tf.Variable(tf.random_uniform([n_hidden2], -1, 1))
encoder2 = tf.nn.sigmoid(tf.add(tf.matmul(encoder1, W2_encode), b2_encode))

W1_decode = tf.Variable(tf.random_uniform([n_hidden2, n_hidden1], -1, 1))
b1_decode = tf.Variable(tf.random_uniform([n_hidden1], -1, 1))
decoder1 = tf.nn.sigmoid(tf.add(tf.matmul(encoder2, W1_decode), b1_decode))

W2_decode = tf.Variable(tf.random_uniform([n_hidden1, n_input], -1, 1))
b2_decode = tf.Variable(tf.random_uniform([n_input], -1, 1))
decoder2 = tf.nn.sigmoid(tf.add(tf.matmul(decoder1, W2_decode), b2_decode))

cost = tf.reduce_mean(tf.square(Y-decoder2))

opt = tf.train.AdamOptimizer(
    learning_rate=learning_rate,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-08,
    use_locking=False,
    name='Adam'
).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_batch = int(mnist.train.num_examples / batch_size)

    for epoch in range(epoch_num):
        avg_cost = 0

        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            batch_xs_noisy = batch_xs + noise_level * np.random.normal(loc=0.0, scale=1.0, size=batch_xs.shape)
            c, _ = sess.run([cost, opt], feed_dict={X: batch_xs_noisy, Y: batch_xs})
            avg_cost += c / total_batch

        print(f'Epoch: {epoch + 1}, cost = {avg_cost: .9f}')

    test_x = mnist.test.images[:10] + noise_level * np.random.normal(loc=0.0, scale=1.0,
                                                                     size=mnist.test.images[:10].shape)

    samples = sess.run(decoder2, feed_dict={X: test_x})
    fig, ax = plt.subplots(3, 10, figsize=(10, 3))

    for i in range(10):
        for j in range(3):
            ax[j][i].set_axis_off()

        ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        ax[1][i].imshow(np.reshape(test_x[i], (28, 28)))
        ax[2][i].imshow(np.reshape(samples[i], (28, 28)))

    plt.show()
