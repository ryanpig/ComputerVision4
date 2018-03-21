import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
import cv2
import glob
import random

Flag_reduce = False
Flag_rotate = True

def normalize(arr):
    if arr.max() > 1.0:
        arr/=255.0
    return arr
def read_from_files():
    gTestFilters = './Data/testing/*.*'
    filelist_Test = glob.glob(gTestFilters)
    imgs = []
    for index, file_path in enumerate(filelist_Test):
        img = mpimg.imread(file_path)[:, :, :3]  # Do not read alpha channel.
        imgs.append(img)
        print(np.shape(imgs))
        print(type(img))
    return imgs

def tf_distort_images(images):
    X_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, (None, None, 3))
    # Configuration
    height = 90  #Crop area
    width = 90
    REDUCE_IMAGE_SIZE = 80  # Reduced size

    if Flag_reduce == True:
        tf_img = tf.image.resize_images(X, (REDUCE_IMAGE_SIZE, REDUCE_IMAGE_SIZE),
                                            tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        tf_img = tf.image.resize_images(X, (120, 120),
                                        tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    else:
        tf_img = tf.image.resize_images(X, (120, 120),
                                        tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Randomly crop a [height, width] section of the image.
    tf_img = tf.random_crop(tf_img, [height, width, 3])
    # Flip Left/Right
    tf_img = tf.image.random_flip_left_right(tf_img)
    # Brightness / Contrast
    tf_img = tf.image.random_brightness(tf_img, max_delta=63)
    tf_img = tf.image.random_contrast(tf_img, lower=0.2, upper=1.8)
    # Rotate Image
    Angle = tf.Variable(initial_value=0, dtype=tf.float32)
    tf_img = tf.contrib.image.rotate(tf_img, Angle, interpolation='NEAREST',
    name=None)
    # Resize back to original size
    tf_img = tf.image.resize_images(tf_img, (120, 120),
                                    tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Standardization
    # tf_img = tf.image.per_image_standardization(tf_img)

    # Session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # loop all images and feed into data flow.
        for i in range(len(images)):
            print(i)
            # Rotate control
            if Flag_rotate == True:
                ang = random.randint(-5,5) # Rotation range
            else:
                ang = 0.0
            img = images[i]
            # Run all processes
            distorted_img = sess.run(tf_img,feed_dict={X:images[i], Angle:ang})
            X_data.append(distorted_img)
    X_data = np.array(X_data, dtype=np.float32)  # Convert to numpy
    return X_data

# Debugging: Make sure distorted images.
tmp = read_from_files()
imgs = tf_distort_images(tmp)

for i in range(len(imgs)):
    normalize(imgs[i])
    cv2.imshow("Show",imgs[i])
    cv2.waitKey(30)
