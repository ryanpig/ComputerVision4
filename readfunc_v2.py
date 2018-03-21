
import tensorflow as tf
import glob
import cv2
import numpy as np
from data_aug import tf_distort_images

# Configuration
Division_ratio = 0.8 # the data ratio for training and evaluating data.

# Define basic data format for training, evaluating, and testing
class examples:
    def __init__(self, len):
        self.labels = np.zeros(len, dtype=np.int32)
        self.images = np.zeros([len,120,120,3], dtype=np.float32)
def normalize(arr):
    if arr.max() > 1.0:
        arr/=255.0
    return arr

def readData():
    # Get file lists from both training and testing folders
    gTrainFilters = './Data/training/*.*'
    filelist = glob.glob(gTrainFilters)
    gTestFilters = './Data/testing/*.*'
    filelist_Test = glob.glob(gTestFilters)

    print("Start to read training data:", gTrainFilters,
          "Testing data:", gTestFilters )
    # labeling and reading features
    data = labeling(filelist)
    test_d = labeling(filelist_Test)
    # Divide data into training and evaluating data
    train_d, eval_d = dataDivision(data, Division_ratio)
    print("testing examples:",len(test_d.labels))
    # Data Augmentation
    # tmp = tf_distort_images(train_d.images)
    # train_d.images =
    return train_d, eval_d, test_d
# labeling based on the file name.
def labeling(filelist):
    lenF = len(filelist)
    examples_d = examples(lenF)
    labels = np.zeros(lenF, dtype=np.int32)
    for i in range(lenF):
        labels[i] = 0 if 'BrushingTeeth' in filelist[i] else labels[i]
        labels[i] = 1 if 'CuttingInKitchen' in filelist[i] else labels[i]
        labels[i] = 2 if 'JumpingJack' in filelist[i] else labels[i]
        labels[i] = 3 if 'Lunges' in filelist[i] else labels[i]
        labels[i] = 4 if 'WallPushups' in filelist[i] else labels[i]
        # 0 for grey scale
        # data.images[i] = cv2.imread(filelist[i],0)
        examples_d.images[i] = cv2.imread(filelist[i])
        # normalization
        normalize(examples_d.images[i])
        examples_d.labels = labels
    return examples_d
def dataDivision(data, division):
    total_len = len(data.labels)
    tr_len = int(total_len * division)
    te_len = total_len - tr_len
    print("training examples:", tr_len, "- evaluate examples:", te_len)
    # initialize train & eval
    train = examples(tr_len)
    evaluate = examples(te_len)
    # shuffle and divide the data
    arr = np.arange(total_len)
    np.random.shuffle(arr)
    for i in range(tr_len):
        ind = arr[i]
        train.labels[i] = data.labels[ind]
        train.images[i] = data.images[ind]
    for k in range(te_len):
        ind = arr[k]
        evaluate.labels[k] = data.labels[ind]
        evaluate.images[k] = data.images[ind]
    return train, evaluate


train1,eval1, test1 = readData()
train_data = train1.images
train_labels = train1.labels
eval_data = eval1.images
eval_labels = eval1.labels
test_data = test1.images
test_labels =  test1.labels
