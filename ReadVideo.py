import cv2
import numpy as np
import glob
# Goal: Get training data -- Middle frame of each video
# Innovation: Pick a promising frame?


# Get five file list in five category.
globFilters = ['./Data/BrushingTeeth/*.*',
               './Data/CuttingInKitchen/*.*',
               './Data/JumpingJack/*.*',
               './Data/Lunges/*.*',
               './Data/WallPushups/*.*'
               ]

# Loop all videos in five categories
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
        # Read middle frame
        mid = int(lenOfVideo / 2)
        # Set to middle frame of video
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, fm = cap.read()
        if ret == True:
            # cv2.imshow('Middle Frame', fm)
            # cv2.waitKey(20)
            # Get the filename & Write the frame into JPG file.
            # dir = curFile.split("v_")[0] + "frames\\"
            dir = ".\Data\\training\\"

            fileName = curFile.split("v_")[1].split(".")[0]
            fileName = fileName + ".jpg"
            path = dir + fileName
            # resize frame to 120x120
            fm_resize = cv2.resize(fm, (120, 120))
            ret2 = cv2.imwrite(path, fm_resize)
            if ret2 == True:
            #   print('write to file:', fileName)
                countWrite = countWrite + 1
    print('Total no. of Write:', countWrite)

# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()
