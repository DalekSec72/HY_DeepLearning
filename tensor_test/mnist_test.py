from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import numpy as np
import matplotlib.pyplot as plt

x,y = mnist.train.next_batch(1)

mnist_image = np.array(x).reshape((28, 28))

plt.title("label: " + str(np.where(y[0] == 1)[0][0]))
plt.imshow(mnist_image, cmap="gray")
plt.show()