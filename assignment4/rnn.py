# -*- coding: utf-8 -*-

# 2021 HYU. CSE
# Taehun Kim <th6424@gmail.com>

import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

sample = 'if you want you'
idx_to_char = list(set(sample))
char_to_idx = {c: i for i, c in enumerate(idx_to_char)}

dict_size = len(char_to_idx)
hidden_size = len(char_to_idx)
num_classes = len(char_to_idx)

batch_size = 1
sequence_length = len(sample) - 1
learning_rate = 0.1

sample_idx = [char_to_idx[c] for c in sample]
x_data = [sample_idx[:-1]]  # if you want yo
y_data = [sample_idx[1:]]  # f you want you

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

x_one_hot = tf.one_hot(X, num_classes)
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=initial_state, dtype=tf.float32)

X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)

outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})

        result_str = [idx_to_char[c] for c in np.squeeze(result)]

        print(i, 'loss:', l, 'Prediction:', ''.join(result_str))
