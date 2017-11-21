import numpy as np
import tensorflow as tf
np.random.seed(0)

# Tensor dimensions
d1, d2 = 4, 5

# Computational graph definition
t1 = tf.placeholder(tf.float32)
t2 = tf.placeholder(tf.float32)
t3 = tf.placeholder(tf.float32)
a = t1 + t2
b = a * t3
c = tf.reduce_sum(b)

# Gradients computation definition
grad_t1, grad_t2, grad_t3 = tf.gradients(c, [t1, t2, t3])

with tf.Session() as sess:
    values = {
        t1: np.random.rand(d1, d2),
        t2: np.random.rand(d1, d2),
        t3: np.random.rand(d1, d2),
    }
    out = sess.run([c, grad_t1, grad_t2, grad_t3], feed_dict=values)

    c_val, grad_t1_val, grad_t2_val, grad_t3_val = out
