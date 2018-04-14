import cv2
import numpy as np
import glob
import random
# Goal: Generate RGB images from both UCF101 videos and my own filmed video.

# Decide to generate RGB images from UCF101 or filmed video.
Flag_training_or_test = False
if Flag_training_or_test == True:
    # Input folder
    globFilters = ['./Data/BrushingTeeth/*.*',
                   './Data/CuttingInKitchen/*.*',
                   './Data/JumpingJack/*.*',
                   './Data/Lunges/*.*',
                   './Data/WallPushups/*.*'
                   ]
    # Output folder
    dir = ".\Data\\training\\"
else:
    # Input folder
    globFilters = ['./Data/Recorded/*.*']
    # Output folder
    dir = ".\Data\\testing\\"

# Loop all videos in five folders or one filmed folder
for k in range(len(globFilters)):
    # Get filelist of a category
    filelist = glob.glob(globFilters[k])
    print('Total no. of videos', len(filelist))
    countWrite = 0
    # Loop each video
    for i in range(len(filelist)):
        curFile = filelist[i]
        cap = cv2.VideoCapture(curFile)
        # print("FileName:", curFile)
        # Check if camera opened successfully
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
        # Get total frames of the video
        lenOfVideo = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print("Total no. of frames", lenOfVideo)
        # Read middle frame +- 0~40% frames.
        random_portion = int(lenOfVideo / 10) * random.randint(-4,4)
        mid =  int(lenOfVideo / 2) + random_portion
        # Set to middle frame of video
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, fm = cap.read()
        if ret == True:
            # Get the filename & Write the frame into JPG file.
            fileName = curFile.split("v_")[1].split(".")[0]
            fileName = fileName + ".jpg"
            path = dir + fileName
            # resize frame to 120x120
            fm_resize = cv2.resize(fm, (120, 120))
            ret2 = cv2.imwrite(path, fm_resize)
            if ret2 == True:
                countWrite = countWrite + 1
    print('Total no. of Write:', countWrite)

# Release the video capture object
cap.release()
# Close all
cv2.destroyAllWindows()
