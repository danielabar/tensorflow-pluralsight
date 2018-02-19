import tensorflow as tf

# Integer, 1D vector with 3 elements
x = tf.placeholder(tf.int32, shape=[3], name='x')
y = tf.placeholder(tf.int32, shape=[3], name='y')

# placeholders can be treated the same as constants
sum_x = tf.reduce_sum(x, name='sum_x')
prod_y = tf.reduce_prod(y, name='prod_y')
final_div = tf.div(sum_x, prod_y, name='final_div')

# run graph
sess = tf.Session()
# `feed_dict` parameter passes in value of placeholder(s)
print('sum(x): ', sess.run(sum_x, feed_dict={x: [100, 200, 300]}))
print('prod(y): ', sess.run(prod_y, feed_dict={y: [1, 2, 3]}))
# This calculation needs both x and y placeholders:
print('sum(x) / prod(y): ', sess.run(final_div, feed_dict={x: [10, 20, 30], y: [1, 2, 3]}))

# write out graph
writer = tf.summary.FileWriter('./logs/m3_example1', sess.graph)
writer.close()
sess.close()