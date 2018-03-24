import numpy as np
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

# Store the MNIST data in ./mnist_data
# First argument `mnist_data/` is where gzip files will be stored
# Second argument defines how labels associated with every image are represented, in this case,
# labels will be 10-element vector, composed of all zeros, except a value of one only at the index which represents the digit.
mnist = input_data.read_data_sets('mnist_data/', one_hot=True)

# Retrieve 5000 training digits and training labels
training_digits, training_labels = mnist.train.next_batch(5000)

# Retrieve 200 test digits and labels
test_digits, test_labels = mnist.test.next_batch(200)

# Setup placeholders for training and test digits
# type of placeholder is float because images are grayscale
# shape of placeholder is [None, 784] - because its  a list of images
# first dimension is index of each image which we don't know how many images will be passed in, so None
# size of each image vector is known: 784
training_digits_pl = tf.placeholder('float', [None, 784])

# Shape of test digit is [784] -> a vector with 784 elements
# Image is 28w x 28h = 784px, each row from grid laid out beside eachother for single vector of 784 elements.
test_digit_pl = tf.placeholder('float', [784])

# Nearest Neighbor calculation using L1 distance
# `tf.negative` flips sign for every element in test digit placeholder
l1_distance = tf.abs(tf.add(training_digits_pl, tf.negative(test_digit_pl)))

distance = tf.reduce_sum(l1_distance, axis=1)

# Prediction: Get min distance index (nearest neighbor)
pred = tf.arg_min(distance, 0)

accuracy = 0

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # loop over test data
    for i in range(len(test_digits)):
        nn_index = sess.run(pred, \
            feed_dict={training_digits_pl: training_digits, test_digit_pl: test_digits[i, :]})

        # use index to find nn class label and compare it to its true label
        # `np.argmax` provides index in one_hot notation
        print('Test', i, 'Prediction: ', np.argmax(training_labels[nn_index]), 'True label', np.argmax(test_labels[i]))

        # Calculate accuracy
        if np.argmax(training_labels[nn_index]) == np.argmax(test_labels[i]):
            accuracy += 1./len(test_digits)

    print('Done!')
    print('Accuracy: ', accuracy)