import tensorflow as tf

# y = Wx + b
W = tf.Variable([2.5, 4.0], tf.float32, name='var_W')
x = tf.placeholder(tf.float32, name='x')
b = tf.Variable([5.0, 10.0], tf.float32, name='var_b')
y = W * x + b

# Initialize all variables defined
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # Computation step to initialize variables
    sess.run(init)
    # Now the variables can be used in computing the graph
    print('Final result: Wx + b = ', sess.run(y, feed_dict={x: [10, 100]}))

# calculte intermediate value, notice this only needs one variable `W`
s = W * x
# Initialize only required variables
init = tf.variables_initializer([W])

with tf.Session() as sess:
    sess.run(init)
    # Note at this point only `W` has been initialized but not `b`. Both are needed to compute `y`, this line will get error:
    # print('Will this work? Wx + b = ', sess.run(y, feed_dict={x: [10, 100]}))

    # This should work to compute `s` because it only requires the variable `W`
    print('Result: Wx = ', sess.run(s, feed_dict={x: [10, 100]}))

# Now let's use variables how they're intended, by updating
# Start with two scalar variables
number = tf.Variable(2)
multiplier = tf.Variable(1)

# initialize all variables
init = tf.global_variables_initializer()

# assign new value to the number variable: number = number * multiplier
# assignment computation is stored in result and can be executed using session run
# `result` is a computation node just like any other in TF
result = number.assign(tf.multiply(number, multiplier))
