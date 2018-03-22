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

        # Transform image - flip upside down
        image = tf.image.flip_up_down(image)

        # Crop a portion of image - eg: central 50%
        # (this will change dimensions of image, in this case, reducing by half)
        image = tf.image.central_crop(image, central_fraction=0.5)

        # set image width, height and number of channels
        image.set_shape((224, 224, 3))

        # Perform the image resize
        image_array = sess.run(image)
        print(image_array.shape)

        # Convert a numpy array of the kind (224, 224, 3) to a Tensor of shape (224, 224, 3)
        image_tensor = tf.stack(image_array)
        print(image_tensor)

        # Image list holds a list of image tensors
        image_list.append(image_tensor)


    # Finish off the filename queue coordinator...
    # Stop all threads
    coord.request_stop()
    # Wait for all threads to complete
    coord.join(threads)

    # Converts all tensors to a single tensor with a 4th dimension
    # 4 images of (224, 224, 3) can be accessed as:
    #   (0, 224, 224, 3), (1, 224, 224, 3), etc.
    images_tensor = tf.stack(image_list)
    print(images_tensor)

    summary_writer = tf.summary.FileWriter('./logs/m4_example3', graph=sess.graph)

    # Write out all the images in one go
    summary_str = sess.run(tf.summary.image('images', images_tensor, max_outputs=4))
    summary_writer.add_summary(summary_str)
    summary_writer.close()