
import tensorflow as tf
import numpy as np
from PIL import Image
#image = tf.image.decode_jpeg("Hist.jpg")
#resized_image = tf.image.resize_images(image, [299, 299])

#print(image)
#print(resized_image)

#  list of files to read
filename_queue = tf.train.string_input_producer(["Hist.jpg", "man.jpg"])


reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)
my_img = tf.image.decode_jpeg(value, channels=3)
#my_img = tf.image.decode_png(value) # use png or jpg decoder based on your files.
print(my_img)
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
  #sess.run(init_op)
  sess.as_default()

  # Start populating the filename queue.

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(2): #length of your filename list
    image = my_img.eval() #here is your image Tensor :)
    print(image.shape)
    Image.fromarray(np.asarray(image)).show()


  #print(image.shape)

  coord.request_stop()
  coord.join(threads)
