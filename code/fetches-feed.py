import tensorflow as tf

# y = Wx + b
# W is a constant 1-D tensor with two values: 10 and 100
W = tf.constant([10, 100], name='const_W')
# x and b are integer placeholders, their values will be provided later when the graph is run
# NOTE that shape has not been specified in placeholder definition, therefore can hold tensors of any shape
x = tf.placeholder(tf.int32, name='x')
b = tf.placeholder(tf.int32, name='b')

# compute Wx
# NOTE `tf.multiply` is simple multiplication, NOT matrix multiplication
# Therefore every element of `W` will be multiplied by every element of `x`
# `W` and `x` must be compatible tensors -> i.e. `x` must be same shape and rank as `W`,
# i.e. vector with two elements
Wx = tf.multiply(W, x, name='Wx')

# compute y
y = tf.add(Wx, b, name='y')

# y_ = x - b
y_ = tf.subtract(x, b, name='y_')

# use with command to work with tf session, will be automatically closed at end of block
with tf.Session() as sess:
    # sess.run(fetches, feed_dict)
    #   fetches: node of graph to be computed
    #   feed_dict: pass in placeholders for computation
    print('Intermediate result: Wx = ', sess.run(Wx, feed_dict={x: [3, 33]}))

    # NOTE value of `x` here is different from intermediate calculation above
    # Every invocation of `sess.run` is an independent calculation of a specified node
    print('Final result: Wx + b = ', sess.run(y, feed_dict={x: [5, 50], b: [7, 9]}))

    # When computing node in graph, don't need to specify every value from scratch,
    # can specify intermediate values, in this case, specify pre-compuuted value of `Wx`
    # this is useful for debugging
    print('Intermediate specified: Wx + b', sess.run(fetches=y, feed_dict={Wx: [100, 1000], b: [7, 9]}))

    # Can also calculate results for multiple nodes using a single session run statement
    # In this case `fetches` is an array of all nodes to be computed
    print('Two results: [Wx + b, x - b] = ', sess.run(fetches=[y, y_], feed_dict={x: [5, 50], b: [7, 9]}))

writer = tf.summary.FileWriter('./logs/m3_example2', sess.graph)
writer.close()
