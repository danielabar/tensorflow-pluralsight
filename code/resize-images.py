import tensorflow as tf
from PIL import Image

original_image_list = [
    './process-images/cute-dog.jpg',
    './process-images/pexels.jpeg',
    './process-images/black-and-white-tree-branches.jpg',
    './process-images/lost-places.jpg'
]

# Make a queue of file names including all the images specified
filename_queue = tf.train.string_input_producer(original_image_list)

# Read an entire image file
image_reader = tf.WholeFileReader()

# use multi-threaded nature of session object to read in multiple image files
with tf.Session() as sess:
    # coordinate the loading of image files
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    image_list = [];
    for i in range(len(original_image_list)):
        # Read a whole file from the queue. First returned value in tuple is filename which we ignore
        _, image_file = image_reader.read(filename_queue)

        # Decode the image as a jpeg file -> convert to Tensor
        # `image` is reference to an image tensor
        image = tf.image.decode_jpeg(image_file)

        # Get a tensor of resized images
        image = tf.image.resize_images(image, [224, 224])

        # set image width, height and number of channels
        image.set_shape((224, 224, 3))

        # Perform the image resize
        image_array = sess.run(image)
        print(image_array.shape)

        # Use Pillow to display resized image
        Image.fromarray(image_array.astype('uint8'), 'RGB').show()

        # expand number of dimensions this tensor has before adding it to image list
        # expand_dims adds a new dimension - we'll use the first dimension to indicate which image is being referenced
        # convenient to have a single tensor represent all images - also good for batch operations
        image_list.append(tf.expand_dims(image_array, 0))

    # Finish off the filename queue coordinator...
    # Stop all threads
    coord.request_stop()
    # Wait for all threads to complete
    coord.join(threads)

    # Write out image summaries to tensor board
    index = 0
    summary_writer = tf.summary.FileWriter('./logs/m4_example2', graph=sess.graph)

    # Write out summary statistics for each image
    for image_tensor in image_list:
        summary_str = sess.run(tf.summary.image('image-' + str(index), image_tensor))
        summary_writer.add_summary(summary_str)
        index += 1

    summary_writer.close()
