import cv2
import numpy as np
import glob
import random

# Goal: Generate optical images from both UCF101 videos and my own filmed video.
# Decide to generate optical flow images from UCF101 or filmed video.

Flag_training_or_test = True
if Flag_training_or_test == True:
    globFilters = ['./Data/BrushingTeeth/*.*',
                   './Data/CuttingInKitchen/*.*',
                   './Data/JumpingJack/*.*',
                   './Data/Lunges/*.*',
                   './Data/WallPushups/*.*'
                   ]
    dir = ".\Data\\training_optical\\"
else:
    globFilters = ['./Data/Recorded/*.*']
    dir = ".\Data\\testing_optical\\"

# Loop all videos in five categories
for k in range(len(globFilters)):
    # Get filelist of a category
    filelist = glob.glob(globFilters[k])
    print('Total no. of videos', len(filelist))
    countWrite = 0
    # Loop each video
    for i in range(len(filelist)):  #len(filelist)
        curFile = filelist[i]
        cap = cv2.VideoCapture(curFile)
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
        # Calculate the position
        lenOfVideo = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        random_portion = int(lenOfVideo / 10) * random.randint(-4,4)
        mid =  int(lenOfVideo / 2) + random_portion
        # Set the position of frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, fm1 = cap.read()
        prvs = cv2.cvtColor(fm1, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(fm1)
        hsv[..., 1] = 255
        # Read the second frame
        ret, fm2 = cap.read()
        next = cv2.cvtColor(fm2, cv2.COLOR_BGR2GRAY)
        # prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags) -> flow
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if ret == True:
            fileName_ori = curFile.split("v_")[1].split(".")[0]
            fileName = fileName_ori + ".jpg"
            fileName_of = fileName_ori + "_of.jpg"
            path = dir + fileName
            path_of = dir + fileName_of
            print(path)
            print(path_of)
            # resize frame to 120x120
            fm_resize = cv2.resize(fm2, (120, 120))
            fm_resize_of = cv2.resize(bgr, (120, 120))
            #ret2 = cv2.imwrite(path, fm_resize)
            # Write single optical flow into jpg image.
            ret3 = cv2.imwrite(path_of, fm_resize_of)
            prvs = next
            if ret3 == True:
                  countWrite = countWrite + 1
    print('Total no. of Write:', countWrite)

# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()
