import tensorflow as tf

# Declare a few 1 dimensional tensors
x = tf.constant([100, 200, 300], name='x')
y = tf.constant([1, 2, 3], name='y')

# TF library has math operations that operate on all elements within a tensor
sum_x = tf.reduce_sum(x, name='sum_x')
prod_y = tf.reduce_prod(y, name='prod_y')
final_div = tf.div(sum_x, prod_y, name='final_div')
# avg
final_mean = tf.reduce_mean([sum_x, prod_y], name='final_mean')

sess = tf.Session()
print('x: ', sess.run(x))
print('y: ', sess.run(y))
print('sum_x: ', sess.run(sum_x))
print('prod_y: ', sess.run(prod_y))
print('final_div: ', sess.run(final_div))
print('final_mean: ', sess.run(final_mean))

writer = tf.summary.FileWriter('./logs/m2_example3', sess.graph)