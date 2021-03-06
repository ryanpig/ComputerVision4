###
# readData(): Read images and return train, eval, and test data sets
# labeling(): Labeling each image based on the filename
# tf_distort_images() in data_aug.py: Data augmentation
# -> flip, reduce, crop, resize, contrast, brightness, translation
# dataDivision() : Divide data into training & evaluation
# Process flow: readData -> labeling -> dataDivision -> tf_distort_images
# 3/28: Modify to Optical Flow version
# training Folder: training_optical
# Example size: 120,120,2

import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from data_aug import tf_distort_images
from sklearn.model_selection import train_test_split
from enum import Enum
from copy import deepcopy

class FeatureType(Enum):
    RGB = 1
    OPTICAL = 2
    OPTICAL_MULTI = 3
    # Get both RGB & OPTICAL_MULTI
    # Not support Data Augmentation.
    Dual = 4


# Configuration
# Feature type: RGB, OpticalFlow, multi OpticalFlow, Dual stream
Global_Feature_Type = FeatureType.Dual

Flag_data_aug = True#Double training data (Only support RGB features)
Flag_Visual = True #Showing sample pictures after feature extraction
Division_ratio = 0.2 # (20% eval), the data ratio for training and evaluating data.
Training_usage_ratio = 0.03 #Decide how much training examples will be used. (0.03 -> 18examples
# Define basic data format for training, evaluating, and testing
class examples:
    def __init__(self, len, feature_type):
        self.labels = np.zeros(len, dtype=np.int32)
        if feature_type == FeatureType.RGB:
            self.images = np.zeros([len,120,120,3], dtype=np.float32)
        elif feature_type == FeatureType.OPTICAL:
            self.images = np.zeros([len,120,120,2], dtype=np.float32)
        elif feature_type == FeatureType.OPTICAL_MULTI:
            self.images = np.zeros([len, 120, 120, 20], dtype=np.float32)
        elif feature_type == FeatureType.Dual:
            self.images = []
            for i in range(len):
                self.images.append(dualFeature())
                #self.images[i].images1 = np.zeros([120,120,3], dtype=np.float32)
                #self.images[i].images2 = np.zeros([120, 120, 20], dtype=np.float32)
    def copy(self):
        a =  examples(len(self.images), FeatureType.RGB)
        a.images = self.images.copy()
        a.labels = self.labels.copy()
        return a
class dualFeature:
    def __init__(self):
        self.images1 = np.zeros([120,120,3], dtype=np.float32) #Spatial feature
        self.images2 = np.zeros([120,120,20], dtype=np.float32)#Temporal feature, 10 frames * (u,v)
def normalize(arr):
    if arr.max() > 1.0:
        arr/=255.0
    return arr
def single_dataset_gen(train_test_ratio, train_eval_ratio,feature_type, flag_data_aug):
    #Train_test_ratio = train_test_ratio
    #Division_ratio = train_eval_ratio

    # Read data and labeling
    traindata,test_d, test_film = readData(feature_type, train_test_ratio)
    # Divide data into training and evaluating data
    train_d, eval_d = dataDivision(traindata, train_eval_ratio, feature_type)
    # Data Augmentation (Double training data:496 -> 992)
    if flag_data_aug:
        # (images, flag_reduce, flag_rotate, flag_mirror,  flag_crop)
        train_d = data_augmentation(train_d,False,False,True,False)
    # Display result
    print("Generate Feature type:",str(feature_type))
    print("Train/Test Usage Ratio:", train_test_ratio,
          "Train/Eval Ratio", train_eval_ratio,
          "Training examples:", len(train_d.labels),
          "Evaluate examples:", len(eval_d.labels),
          "Testing_Hold examples:", len(test_d.labels),
          "Testing_Film examples:", len(test_film.labels))
    return train_d, eval_d, test_d, test_film

# Only return train & eval data.
def kfold_dataset_gen(feature_type, train_test_ratio):
    # Read data and labeling
    traindata, test_d, test_film = readData(feature_type, train_test_ratio)
    return traindata, test_d, test_film
def readData(feature_type, train_test_ratio):
    # Get file lists from both training and testing folders
    if feature_type == feature_type.RGB:
        gTrainFilters = './Data/training/*.*'
        gFilmFilters = './Data/testing/*.*'
    elif feature_type == feature_type.OPTICAL:
        gTrainFilters = './Data/training_optical/*.*'
        gFilmFilters = './Data/testing_optical/*.*'
    elif feature_type == feature_type.OPTICAL_MULTI: #only as a file list to retrieve multi OFs
        gTrainFilters = './Data/training_optical/*.*'
        gFilmFilters = './Data/testing_optical/*.*'
    elif feature_type == feature_type.Dual: #only as file list to retrieve multi OFs
        gTrainFilters = './Data/training_optical/*.*'
        gFilmFilters = './Data/testing_optical/*.*'

    filelist = glob.glob(gTrainFilters)
    filelist_film = glob.glob(gFilmFilters)
    print("Start to read training data:", gTrainFilters,
          "Filmed data:", gFilmFilters )
    # labeling and reading features

    filelist_Hold, filelist = train_test_split(filelist, test_size=train_test_ratio,
                         shuffle=True)
    data = labeling(filelist, feature_type)
    test_hold = labeling(filelist_Hold, feature_type)
    test_film = labeling(filelist_film, feature_type)

    return data, test_hold, test_film

# labeling based on the file name.
def labeling(filelist, feature_type):
    lenF = len(filelist)
    labels = np.zeros(lenF, dtype=np.int32)
    examples_d = examples(lenF, feature_type)
    for i in range(lenF):
        # Labeling
        labels[i] = 0 if 'BrushingTeeth' in filelist[i] else labels[i]
        labels[i] = 1 if 'CuttingInKitchen' in filelist[i] else labels[i]
        labels[i] = 2 if 'JumpingJack' in filelist[i] else labels[i]
        labels[i] = 3 if 'Lunges' in filelist[i] else labels[i]
        labels[i] = 4 if 'WallPushups' in filelist[i] else labels[i]
        examples_d.labels = labels
        # Feature Extraction
        if feature_type == FeatureType.RGB:
            #examples_d.images[i] = cv2.cvtColor(cv2.imread(filelist[i]),cv2.COLOR_BGR2RGB)
            examples_d.images[i] = plt.imread(filelist[i])
            # Normalization
            normalize(np.asarray(examples_d.images[i], dtype=np.float32))
        elif feature_type == FeatureType.OPTICAL:
            # HSV -> H,V -> optical flow U,V
            img = cv2.cvtColor(cv2.imread(filelist[i]),cv2.COLOR_BGR2HSV)
            examples_d.images[i][...,0] = img[...,0]
            examples_d.images[i][...,1] = img[...,2]
            # Normalization
            normalize(np.asarray(examples_d.images[i], dtype=np.float32))
        elif feature_type == FeatureType.OPTICAL_MULTI:
            dir_train = './Data\\training_optical_multi\\'  # In MacOS, the '\\' has to be replaced to '/'
            dir_test = './Data\\testing_optical_multi\\'   # And, also change split function as below
            for n in range(10):
                # ./Data/training_optical\\BrushingTeeth_g01_c01_of.jpg
                filename = filelist[i].split('_of')[0].split('\\')[1]   # ('\\')[1] -> ('/')[-1]
                dir_type = filelist[i].split('Data/')[1].split('_')[0]
                print(filename)
                #print(dir_type)
                if dir_type == 'training':
                    path = dir_train + filename + '_of_' + str(n) + '.jpg'
                elif dir_type == 'testing':
                    path = dir_test + filename + '_of_' + str(n) + '.jpg'
                print(path)
                img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2HSV)
                #tmp = np.zeros(shape=(120,120,2))
                examples_d.images[i][..., 2*n] = img[..., 0]
                examples_d.images[i][..., (2*n+1)] = img[..., 2]
                # Normalization
                normalize(np.asarray(examples_d.images[i], dtype=np.float32))
        elif feature_type == FeatureType.Dual:
            #Spatio features (RGB) (120,120,3)
            filename = filelist[i].split('_of')[0].split('\\')[1]  # In MacOS, the '\\' has to be replaced to '/'
            dir_type = filelist[i].split('Data/')[1].split('_')[0]    # ('\\')[1] -> ('/')[-1]
            dir_train = './Data\\training\\'        # In MacOS, the '\\' has to be replaced to '/'
            dir_test = './Data\\testing\\'          # In MacOS, the '\\' has to be replaced to '/'
            if dir_type == 'training':
                path = dir_train + filename + '.jpg'
            elif dir_type == 'testing':
                path = dir_test + filename + '.jpg'
            examples_d.images[i].images1 =  np.asarray(plt.imread(path),dtype=np.float32)

            normalize(examples_d.images[i].images1)

            #Motion Features (Optical flow) (120,120,20)
            dir_train = './Data\\training_optical_multi\\'  # In MacOS, the '\\' has to be replaced to '/'
            dir_test = './Data\\testing_optical_multi\\'    # In MacOS, the '\\' has to be replaced to '/'
            for n in range(10):
                # ./Data/training_optical\\BrushingTeeth_g01_c01_of.jpg
                filename = filelist[i].split('_of')[0].split('\\')[1]  # ('\\')[1] -> ('/')[-1]
                dir_type = filelist[i].split('Data/')[1].split('_')[0] #training
               # print(filename)
               # print(dir_type)
                if dir_type == 'training':
                    path = dir_train + filename + '_of_' + str(n) + '.jpg'
                elif dir_type == 'testing':
                    path = dir_test + filename + '_of_' + str(n) + '.jpg'
                print(path)
                img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2HSV)
                # tmp = np.zeros(shape=(120,120,2))
                examples_d.images[i].images2[..., 2 * n] = img[..., 0]
                examples_d.images[i].images2[..., (2 * n + 1)] = img[..., 2]
                normalize(np.asarray(examples_d.images[i].images2,dtype=np.float32))
            #print("Temporal feature:")
            #print(np.shape(examples_d.images))
    return examples_d
def dataDivision(data, division_ratio, feature_type):
    total_len = len(data.labels)
    tr_len = int(total_len * division_ratio)
    te_len = total_len - tr_len

    # initialize train & eval
    train = examples(tr_len, feature_type)
    evaluate = examples(te_len, feature_type)
    # shuffle and divide the data
    #if feature_type == feature_type.Dual:
    #    pass
    #else:
    train.images, evaluate.images, train.labels, evaluate.labels = \
        train_test_split(data.images, data.labels,
                         test_size=division_ratio,
                         shuffle=True)

    return train, evaluate

# To do list : Add translation from data_aug_translation.py
def data_augmentation(train_d,flag_reduce, flag_rotate, flag_mirror, flag_crop):
    distorted_images = tf_distort_images(train_d.images, flag_reduce, flag_rotate, flag_mirror, flag_crop)

    for i in range(5):
        #print(sum(distorted_images[i]))
        np.array((distorted_images[i]), dtype=np.float32)
        distorted_images[i] = cv2.cvtColor(distorted_images[i], cv2.COLOR_RGB2BGR)
    train_d.images = np.concatenate((train_d.images, distorted_images), axis=0)
    train_d.labels = np.concatenate((train_d.labels, train_d.labels), axis=0)
    return train_d
def debug():
    feature_type1 = FeatureType.RGB
    train1, eval1, test1, test_film = single_dataset_gen(train_test_ratio=0.8, train_eval_ratio=0.1,
                                                         feature_type=feature_type1, flag_data_aug=True)
    train_data = train1.images
    train_labels = train1.labels
    eval_data = eval1.images
    eval_labels = eval1.labels
    test_data = test1.images
    test_labels =  test1.labels
    print("train------------")
    print(train_data)
    print(np.shape(train_data))
    print("test------------")
    print(test_data)
    print(np.shape(test_data))
    # Show RGB images
    if Flag_Visual == True:
        for i in range(5):
            fig, [(ax1, ax2, ax3), (ax4, ax5, ax6)] = plt.subplots(2, 3, figsize=(8, 4), sharex=True, sharey=True)
            ax1.imshow(train_data[i])
            ax2.imshow(train_data[i+int(len(train_data)/2)+5]) #For data aug. image.
            ax3.imshow(eval_data[i])
            ax4.imshow(eval_data[i+1])
            ax5.imshow(test_data[i])
            ax6.imshow(test_data[i+1])
            str1 = "count" + str(i)
            plt.title(str)
            plt.show()
def debug_dual():
    feature_type1 = FeatureType.RGB
    train1, eval1, test1, test_film = single_dataset_gen(train_test_ratio=0.8, train_eval_ratio=0.1,feature_type=feature_type1)
    train_data = train1.images
    train_labels = train1.labels
    eval_data = eval1.images
    eval_labels = eval1.labels
    test_data = test1.images
    test_labels = test1.labels
    test_film_data = test_film.images
    test_film_labels = test_film.labels
    print("train------------")
    #print(train_data[0].images1)
    #print(train_data[0].images2)
    #print(np.shape(train_data))
    #print(np.shape(train_data[0].images1))
    #print(np.shape(train_data[0].images2))
    #print(np.sum(train_data[0].images1))
    #print(np.sum(train_data[0].images2))

    if Flag_Visual == True: # only support RGB & Dual(RGB)
        for i in range(3):
            fig, [(ax1, ax2, ax3), (ax4, ax5, ax6)] = plt.subplots(2, 3, figsize=(8, 4), sharex=True, sharey=True)
            if feature_type1 == FeatureType.Dual:
                ax1.imshow(train_data[i].images1)
                ax2.imshow(train_data[i+1].images1)  # For data aug. image.
                ax3.imshow(eval_data[i].images1)
                ax4.imshow(eval_data[i + 1].images1)
                ax5.imshow(test_data[i].images1)
                ax6.imshow(test_film_data[i].images1)
            else:
                ax1.imshow(train_data[i])
                ax2.imshow(train_data[i + 1])  # For data aug. image.
                ax3.imshow(eval_data[i])
                ax4.imshow(eval_data[i + 1])
                ax5.imshow(test_data[i])
                ax6.imshow(test_film_data[i])
            str1 = "count" + str(i)
            plt.title(str1)
            plt.show()
# data = kfold_dataset_gen()


#debug()
#debug_dual()