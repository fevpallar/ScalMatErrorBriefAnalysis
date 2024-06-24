

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


x_set = np.linspace(0, 10, 100)
y_set = x_set + np.random.normal(0, 1, 100)


x_to_column = np.transpose(np.matrix(x_set))
that_1_column = np.transpose(np.matrix(np.repeat(1, 100)))

A = np.column_stack((x_to_column, that_1_column))
b = np.transpose(np.matrix(y_set))


A_tensor = tf.constant(A, dtype=tf.float32)
b_tensor = tf.constant(b, dtype=tf.float32)


tA_A = tf.matmul(tf.transpose(A_tensor), A_tensor)
tA_A_inv = tf.linalg.inv(tA_A)
product = tf.matmul(tA_A_inv, tf.transpose(A_tensor))
solution = tf.matmul(product, b_tensor)

coefficients = solution.numpy()
c1 = coefficients[0][0]
c0 = coefficients[1][0]

print('gradient: ' + str(c1))
print('y-intercept: ' + str(c0))


passthrough = []
for i in x_set:
    passthrough.append(c1 * i + c0)

# plotting
plt.plot(x_set, y_set, 'o', label='set..')
plt.plot(x_set, passthrough, 'r-', label='passthrough', linewidth=3)
plt.legend(loc='upper right')
plt.show()
