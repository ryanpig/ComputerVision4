
import tensorflow as tf
import glob
import cv2
import  numpy as np

# Configuration
Division_ratio = 0.8 # the data ratio for training and evaluating data.

# Training class
class tmp:
    def __init__(self):
        self.labels = 0
        self.images = 0 # np.zeros([-1,120,120,3],dtype=np.uint8)
class data:
    def __init__(self):
        self.labels = 0
        self.images = 0 # np.zeros([-1,120,120,3],dtype=np.uint8)
class examples:
    def __init__(self, len):
        self.labels = np.zeros(len, dtype=np.int32)
        self.images = np.zeros([len,120,120,3], dtype=np.float32)

class train:
    def __init__(self):
        self.labels = 0
        self.images = 0 # np.zeros([-1,120,120,3],dtype=np.uint8)

class eval:
    def __init__(self):
        self.labels = 0
        self.images = 0 # np.zeros([-1,120,120,3],dtype=np.uint8)

class test:
    def __init__(self):
        self.labels = 0
        self.images = 0 # np.zeros([-1,120,120,3],dtype=np.uint8)

def normalize(arr):
    #arr=arr.astype('float32')
    if arr.max() > 1.0:
        arr/=255.0
    return arr

def ReadData():
    # Get file lists from both training and testing folders
    gTrainFilters = './Data/training/*.*'
    filelist = glob.glob(gTrainFilters)
    gTestFilters = './Data/testing/*.*'
    filelist_Test = glob.glob(gTestFilters)

    lenTr = len(filelist)
    labels = np.zeros(lenTr, dtype=np.int32)
    data.images = np.zeros([lenTr,120,120,3])

    # labeling
    for i in range(lenTr):
        labels[i] = 0 if 'BrushingTeeth' in filelist[i] else labels[i]
        labels[i] = 1 if 'CuttingInKitchen' in filelist[i]  else labels[i]
        labels[i] = 2 if 'JumpingJack' in filelist[i] else labels[i]
        labels[i] = 3 if 'Lunges' in filelist[i] else labels[i]
        labels[i] = 4 if 'WallPushups' in filelist[i] else labels[i]
        # 0 for grey scale
        # data.images[i] = cv2.imread(filelist[i],0)
        data.images[i] = cv2.imread(filelist[i])
        normalize(data.images[i])
    data.labels = labels
    # deter
    train_d, eval_d = DataDivision(data,Division_ratio)
    return train_d, eval_d

def DataDivision(data, division):
    total_len = len(data.labels)
    tr_len = int(total_len * division)
    te_len = total_len - tr_len
    print("training examples:", tr_len, "- evaluate examples:", te_len)
    # initialize train & eval
    train.images =  np.zeros([tr_len,120,120,3],dtype=np.float32)
    eval.images =  np.zeros([te_len,120,120,3], dtype=np.float32)
    train.labels = np.zeros([tr_len])
    eval.labels = np.zeros([te_len])
    # shuffle and divide the data
    arr = np.arange(total_len)
    np.random.shuffle(arr)
    for i in range(tr_len):
        ind = arr[i]
        train.labels[i] = data.labels[ind]
        train.images[i] = data.images[ind]
    for k in range(te_len):
        ind = arr[k]
        eval.labels[k] = data.labels[ind]
        eval.images[k] = data.images[ind]
    return train, eval


train1,eval1 = ReadData()
train_data = train1.images  # Returns np.array
train_labels = np.asarray(train1.labels, dtype=np.int32)
eval_data = eval1.images  # Returns np.array
eval_labels = np.asarray(eval1.labels, dtype=np.int32)
#test_data =

