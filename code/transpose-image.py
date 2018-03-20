import tensorflow as tf
import matplotlib.image as mp_img
import matplotlib.pyplot as plot
import os

filename = './process-images/DandelionFlower.jpg'

# read in image as a numpy array
image = mp_img.imread(filename)
print('Image shape: ', image.shape) # (1413, 1765, 3)
# print('Image array: ', image)

# display image on screen
# plot.imshow(image)
# plot.show()

# setup `x` as a tf variable that holds the entire image as a tensor
x = tf.Variable(image, name='x')

# init variables
init = tf.global_variables_initializer()

# start session for image operations
with tf.Session() as sess:
    sess.run(init)

    # run transpose operation to flip width and height of image
    # tf.transpose works on any n-dimensional matrix
    # flips any axis of any matrix in any order specified
    # Original axis indexes are: 0, 1, 2
    # New order 1, 0, 2 -> first and second axis are swapped
    # `perm` specifies new axis orders
    # in this case, width and height are swapped, but 3rd axis (pixel values) are not modified

    # generic
    # transpose = tf.transpose(x, perm=[1, 0, 2])
    # specific for images, simpler api
    transpose = tf.image.transpose_image(x)

    # invoke the transpose operation
    result = sess.run(transpose)

    print('Transposed image shape: ', result.shape)
    plot.imshow(result)
    plot.show()