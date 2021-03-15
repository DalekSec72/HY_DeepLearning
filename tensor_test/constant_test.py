import tensorflow as tf

constant_value = tf.constant("deep learning course")
print(constant_value)

ten = tf.constant(10)
nine = tf.constant(9)
nineteen = tf.add(ten, nine)
print(nineteen)

constant_arr = tf.constant([1, 2])
print(constant_arr)

print("===================================")
sess = tf.Session()
print(sess.run(constant_value))
print(sess.run([ten, nine, nineteen]))
print(sess.run(constant_arr))

sess.close()