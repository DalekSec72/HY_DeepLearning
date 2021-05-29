import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn_cell


def batch_data(shuffled_idx, batch_size, data, labels, start_idx):
    idx = shuffled_idx[start_idx:start_idx + batch_size]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def get_vocabulary_size(X):
    return max([max(x) for x in X]) + 1  # plus the 0th word


def fit_in_vocabulary(X, voc_size):
    return [[w for w in x if w < voc_size] for x in X]


def zero_pad(X, seq_len):
    return np.array([x[:seq_len - 1] + [0] * max(seq_len - len(x), 1) for x in X])


def build_classifier(x, vocabulary_size, EMBEDDING_DIM, HIDDEN_SIZE):
    # Embedding layer
    embeddings_var = tf.Variable(tf.random_uniform([vocabulary_size, EMBEDDING_DIM], -1.0, 1.0), trainable=True)
    batch_embedded = tf.nn.embedding_lookup(embeddings_var, x)

    """
    # RNN layer
    with tf.variable_scope('encode'):
        enc_cell = rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        enc_cell = rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=KEEP_PROB)

        enc_outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, batch_embedded, dtype=tf.float32)

    with tf.variable_scope('decode'):
        dec_cell = rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        dec_cell = rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=KEEP_PROB)

        rnn_outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, batch_embedded, initial_state=enc_states,
                                                    dtype=tf.float32)

    concat = tf.concat([enc_states, dec_states], axis=-1)
    with tf.name_scope('v'):
        v = tf.tanh(tf.tensordot(concat, w_omega, axes=1) + b_omega)

    vu = tf.tensordot(v, u_omega, axes=1, name='vu')
    alphas = tf.nn.softmax(vu)
    output = tf.reduce_sum(batch_embedded * tf.expand_dims(alphas, -1), 1)
    """

    # RNN layer
    rnn_outputs, states = tf.nn.dynamic_rnn(rnn_cell.GRUCell(HIDDEN_SIZE), batch_embedded, dtype=tf.float32)

    # Fully connected layer
    # X_for_fc = tf.reshape(rnn_outputs, [-1, HIDDEN_SIZE])
    # print(rnn_outputs, states, X_for_fc)
    W = tf.Variable(tf.random_uniform([HIDDEN_SIZE, 2], -1.0, 1.0), trainable=True)
    b = tf.Variable(tf.random_uniform([2], -1.0, 1.0), trainable=True)

    logits = tf.nn.bias_add(tf.matmul(states, W), b)
    # logits = tf.layers.dense(output, 32, activation='relu')
    # tf.reshape(logits, [-1, 32])
    # logits = tf.layers.dense(logits, 2)
    hypothesis = tf.nn.softmax(logits)

    return hypothesis, logits


ckpt_path = "output/"

SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 100
HIDDEN_SIZE = 150
BATCH_SIZE = 256
NUM_EPOCHS = 25
learning_rate = 0.001
KEEP_PROB = 0.5
ATTENTION_SIZE = 64

# Load the data set
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

x_train = np.load("data/x_train.npy")
y_train = np.load("data/y_train.npy")
x_test = np.load("data/x_test.npy")

np.load = np_load_old

dev_num = len(x_train) // 4

x_dev = x_train[:dev_num]
y_dev = y_train[:dev_num]

x_train = x_train[dev_num:]
y_train = y_train[dev_num:]

y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 2))
y_dev_one_hot = tf.squeeze(tf.one_hot(y_dev, 2))

# Sequences pre-processing
vocabulary_size = get_vocabulary_size(x_train)
x_dev = fit_in_vocabulary(x_dev, vocabulary_size)
x_train = zero_pad(x_train, SEQUENCE_LENGTH)
x_dev = zero_pad(x_dev, SEQUENCE_LENGTH)

batch_ph = tf.placeholder(tf.int32, [None, SEQUENCE_LENGTH], name='batch_ph')
target_ph = tf.placeholder(tf.float32, [None, 2], name='target_ph')

y_pred, logits = build_classifier(batch_ph, vocabulary_size, EMBEDDING_DIM, HIDDEN_SIZE)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target_ph, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Accuracy metric
is_correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(target_ph, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

total_batch = int(len(x_train) / BATCH_SIZE) if len(x_train) % BATCH_SIZE == 0 else int(len(x_train) / BATCH_SIZE) + 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("학습시작")

    for epoch in range(NUM_EPOCHS):
        print("Epoch", epoch + 1)
        start = 0
        shuffled_idx = np.arange(0, len(x_train))
        np.random.shuffle(shuffled_idx)

        for i in range(total_batch):
            batch = batch_data(shuffled_idx, BATCH_SIZE, x_train, y_train_one_hot.eval(), i * BATCH_SIZE)
            sess.run(optimizer, feed_dict={batch_ph: batch[0], target_ph: batch[1]})
        saver = tf.train.Saver()
        saver.save(sess, ckpt_path)
        saver.restore(sess, ckpt_path)

    dev_accuracy = accuracy.eval(feed_dict={batch_ph: x_dev, target_ph: np.asarray(y_dev_one_hot.eval())})
    print("dev 데이터 Accuracy: %f" % dev_accuracy)

    # 밑에는 건드리지 마세요
    x_test = fit_in_vocabulary(x_test, vocabulary_size)
    x_test = zero_pad(x_test, SEQUENCE_LENGTH)

    test_logits = y_pred.eval(feed_dict={batch_ph: x_test})
    np.save("result", test_logits)
