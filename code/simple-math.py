import tensorflow as tf

# declare some constants having float data type
a = tf.constant(6.5, name='constant_a')
b = tf.constant(3.4, name='constant_b')
c = tf.constant(63.0, name='constant_c')
d = tf.constant(100.2, name='constant_d')

# declare some computations nodes
square = tf.square(a, name='square_a')
power = tf.pow(b, c, name='pow_b_c')
sqrt = tf.sqrt(d, name='sqrt_d')
final_sum = tf.add_n([square, power, sqrt], name='final_sum')
another_sum = tf.add_n([a, b, c, d, power], name='another_sum')

# run graph
sess = tf.Session()
print('Square of a: ', sess.run(square))
print('Power of b ^ c: ', sess.run(power))
print('Square root of d: ', sess.run(sqrt))
print('Sum of square, power and square root: ', sess.run(final_sum))

# write out graph to log file so it can be visualized with TensorBoard
writer = tf.summary.FileWriter('./logs/m2_example2', sess.graph)
writer.close()

# session is automatically closed when program terminates but good practice to do so explicitly
sess.close()
