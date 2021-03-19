# -*- coding: utf-8 -*-

# 2020 HYU. CSE
# Taehun Kim <th6424@gmail.com>

import tensorflow as tf

x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0], [1], [1], [0]]

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])


W1 = tf.Variable(tf.random_uniform([2, 2], -1, 1))
b1 = tf.Variable(tf.random_uniform([2], -1, 1))
layer1 = tf.nn.sigmoid(tf.add(tf.matmul(X, W1), b1))

W2 = tf.Variable(tf.random_uniform([2, 1], -1, 1))
b2 = tf.Variable(tf.random_uniform([1], -1, 1))
layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, W2), b2))

output = layer2

cost = tf.reduce_mean(tf.square(output - Y))
opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = opt.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        for x, y in zip(x_data, y_data):
            _, cost_val = sess.run([train_op, cost], feed_dict={X: [x], Y: [y]})

        print(step, cost_val, sess.run(W2), sess.run(b2)) if step % 1000 == 0 else 0

    print(sess.run(output, feed_dict={X: x_data}))
