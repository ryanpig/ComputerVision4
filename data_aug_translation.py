from math import ceil, floor
from data_aug import tf_distort_images
from data_aug import normalize
from data_aug import read_from_files
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import random
IMAGE_SIZE = 120
tran_percentL = 0.2
tran_percentH = 0.8
Flag_translate = True

def get_translate_parameters(index):
    if index == 0:  # Translate left 20 percent
        offset = np.array([0.0, tran_percentL], dtype=np.float32)
        size = np.array([IMAGE_SIZE, ceil(tran_percentH * IMAGE_SIZE)], dtype=np.int32)
        w_start = 0
        w_end = int(ceil(tran_percentH * IMAGE_SIZE))
        h_start = 0
        h_end = IMAGE_SIZE
    elif index == 1:  # Translate right 20 percent
        offset = np.array([0.0, -1 * tran_percentL], dtype=np.float32)
        size = np.array([IMAGE_SIZE, ceil(tran_percentH * IMAGE_SIZE)], dtype=np.int32)
        w_start = int(floor((1 - tran_percentH) * IMAGE_SIZE))
        w_end = IMAGE_SIZE
        h_start = 0
        h_end = IMAGE_SIZE
    elif index == 2:  # Translate top 20 percent
        offset = np.array([tran_percentL, 0.0], dtype=np.float32)
        size = np.array([ceil(tran_percentH * IMAGE_SIZE), IMAGE_SIZE], dtype=np.int32)
        w_start = 0
        w_end = IMAGE_SIZE
        h_start = 0
        h_end = int(ceil(tran_percentH * IMAGE_SIZE))
    else:  # Translate bottom 20 percent
        offset = np.array([-1 *tran_percentL, 0.0], dtype=np.float32)
        size = np.array([ceil(tran_percentH * IMAGE_SIZE), IMAGE_SIZE], dtype=np.int32)
        w_start = 0
        w_end = IMAGE_SIZE
        h_start = int(floor((1 - tran_percentH) * IMAGE_SIZE))
        h_end = IMAGE_SIZE

    return offset, size, w_start, w_end, h_start, h_end


def translate_images(X_imgs, mode):
    offsets = np.zeros((len(X_imgs), 2), dtype=np.float32)
    n_translations = 4
    X_translated_arr = []

    tf.reset_default_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        X_translated = np.zeros((len(X_imgs), IMAGE_SIZE, IMAGE_SIZE, 3),
                                dtype=np.float32)
        X_translated.fill(1.0)  # Filling background color
        base_offset, size, w_start, w_end, h_start, h_end = get_translate_parameters(mode)
        offsets[:, :] = base_offset
        glimpses = tf.image.extract_glimpse(X_imgs, size, offsets)

        glimpses = sess.run(glimpses)
        X_translated[:, h_start: h_start + size[0], \
        w_start: w_start + size[1], :] = glimpses
        X_translated_arr.extend(X_translated)
    X_translated_arr = np.array(X_translated_arr, dtype=np.float32)
    return X_translated_arr

# Translation test
tmp = read_from_files()
imgs = tf_distort_images(tmp)

if Flag_translate == True:
    le = len(imgs)
    batch_size = 2
    le2 = int(le / batch_size) - 1
    print(le2)
    for i in range(le2):
        a = i*batch_size
        b = i*batch_size+batch_size
        c = random.randint(0,3)
        print(c)
        imgs[a:b] = translate_images(imgs[a:b],random.randint(0,3))

# Show
for i in range(le):
    normalize(imgs[i])
    cv2.imshow("Show",imgs[i])
    cv2.waitKey(300)


