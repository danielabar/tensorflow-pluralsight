import tensorflow as tf

# explicitly instantiate a graph
g1 = tf.Graph()

# add tensors and operators to g1 graph
with g1.as_default():
    with tf.Session() as sess:
        # y = Ax + b
        A = tf.constant([5, 7], tf.int32, name='A')
        x = tf.placeholder(tf.int32, name='x')
        b = tf.constant([3,4], tf.int32, name='b')
        y = A * x + b

        # execute y
        print(sess.run(y, feed_dict={x: [10, 100]}))

        # y operator has graph associated with it
        assert y.graph is g1

# instantiate another graph
g2 = tf.Graph()
with g2.as_default():
    with tf.Session() as sess:
        # y = A^x
        A = tf.constant([5, 7], tf.int32, name='A')
        x = tf.placeholder(tf.int32, name='x')
        y = tf.pow(A, x, name='y')
        print(sess.run(y, feed_dict={x: [3, 5]}))
        # this assertion should fail
        # assert y.graph is g1
        # this assertion should pass
        assert y.graph is g2

# when no graphs are explicitly instantiated, all computations are added to the default graph
# get a handle to the default graph:
default_graph = tf.get_default_graph()
# note no `with` statement needed when working with the default graph
with tf.Session() as sess:
    # y = A + x
    A = tf.constant([5, 7], tf.int32, name='A')
    x = tf.placeholder(tf.int32, name='x')
    y = A + x
    print(sess.run(y, feed_dict={x: [3, 5]}))
    assert y.graph is default_graph