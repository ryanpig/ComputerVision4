
import tensorflow as tf
import glob
import cv2
import  numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
# Training class
class tmp:
    def __init__(self):
        self.labels = 0
        self.images = 0 # np.zeros([-1,120,120,3],dtype=np.uint8)
class data:
    def __init__(self):
        self.labels = 0
        self.images = 0 # np.zeros([-1,120,120,3],dtype=np.uint8)
class train:
    def __init__(self):
        self.labels = 0
        self.images = 0 # np.zeros([-1,120,120,3],dtype=np.uint8)

class test:
    def __init__(self):
        self.labels = 0
        self.images = 0 # np.zeros([-1,120,120,3],dtype=np.uint8)

def ReadData(unused_argv):
    # Get filelist of training data set
    gTraingFilters = './Data/training/*.*'
    t = './Data/BrushingTeeth/*.*'
    filelist = glob.glob(gTraingFilters)
    lenTr = len(filelist)
    labels = np.zeros(lenTr, dtype=np.int32)
    data.images = np.zeros([lenTr,120,120])

    # labeling
    for i in range(lenTr):
        labels[i] = 0 if 'BrushingTeeth' in filelist[i] else labels[i]
        labels[i] = 1 if 'CuttingInKitchen' in filelist[i]  else labels[i]
        labels[i] = 2 if 'JumpingJack' in filelist[i] else labels[i]
        labels[i] = 3 if 'Lunges' in filelist[i] else labels[i]
        labels[i] = 4 if 'WallPushups' in filelist[i] else labels[i]
        data.images[i] = cv2.imread(filelist[i],0)

    data.labels = labels
    train_d, test_d = DataDivision(data,0.8)
    return train_d, test_d

def DataDivision(data, division):
    total_len = len(data.labels)
    tr_len = int(total_len * division)
    te_len = total_len - tr_len
    print("training examples:", tr_len, "- testing examples:", te_len)
    # initialize train & test
    train.images =  np.zeros([tr_len,120,120],dtype=np.float16)
    test.images =  np.zeros([te_len,120,120], dtype=np.float16)
    train.labels = np.zeros([tr_len])
    test.labels = np.zeros([te_len])
    # shuffle and divide the data
    arr = np.arange(total_len)
    np.random.shuffle(arr)
    for i in range(tr_len):
        ind = arr[i]
        train.labels[i] = data.labels[ind]
        train.images[i] = data.images[ind]
    for k in range(te_len):
        ind = arr[k]
        test.labels[k] = data.labels[ind]
        test.images[k] = data.images[ind]
    return train, test

un = 1
train1,test1 = ReadData(un)
train_data = train1.images  # Returns np.array
train_labels = np.asarray(train1.labels, dtype=np.int32)
eval_data = test1.images  # Returns np.array
eval_labels = np.asarray(test1.labels, dtype=np.int32)

'''
#  list of files to read
filename_queue = tf.train.string_input_producer(filelist)
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
  for i in range(lenTr): #length of your filename list
    image = my_img.eval() #here is your image Tensor :)
    print(image.shape)
    #Image.fromarray(np.asarray(image)).show()


  #print(image.shape)

  coord.request_stop()
  coord.join(threads)
'''

#train_data = mnist.train.images  # Returns np.array
#train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
#eval_data = mnist.test.images  # Returns np.array
#eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

