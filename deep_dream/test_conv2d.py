import tensorflow as tf
import numpy as np


k = np.float32([1, 4, 6, 4, 1])
k = np.outer(k, k)
k5x5 = k[:, :, None, None] * np.eye(3, dtype=np.float32)
# print(k5x5.shape[:])

img = np.ones([1, 3, 10, 10])
lo = tf.nn.conv2d(img, k5x5, [1, 2, 2, 1], 'SAME')
lo2 = tf.nn.conv2d_transpose(img, k5x5, tf.shape(img), [1, 2, 2, 1])

sess = tf.Session()
print(sess.run(lo2))