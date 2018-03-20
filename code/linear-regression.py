import tensorflow as tf

# Model parameters
# For linear regression we want to find the value of the slope `W` and the y-intercept `b` that give the best fit line
W = tf.Variable([.3],dtype=tf.float32)
b = tf.Variable([-.3],dtype=tf.float32)

# Model input and output
# Want to feed in a range of values for x in training data
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# Initialize placeholder to hold output values of training data
# These are the `y` values corresponding to `x` values of training data
# ML model will use this to train itself
y = tf.placeholder(tf.float32)

# loss: best fit line is one which minimizes the least square error
loss = tf.reduce_sum(tf.square(linear_model - y))

# optimizer: to minimize loss
optimizer = tf.train.GradientDescentOptimizer(0.01)

train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# initialize variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # training data is passed in batches through the model
    for i in range(1000):
        sess.run(train, {x: x_train, y: y_train})

    # evaluate training accuracy
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})

    print('W: %s, b: %s, loss: %s'% (curr_W, curr_b, curr_loss))