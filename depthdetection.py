import cv2
import numpy as numpy
import time

import camera as cam

loadPhotos = True
saveImages = True

imgPath = "FotosStereoVision/"


downScaleSize = (900, 700)

if loadPhotos:
    leftImg = cv2.imread(imgPath + "Left.jpg")
    rightImg = cv2.imread(imgPath + "Right.jpg")


else:
    print("taking left photo")
    leftImg = cam.photo()

    time.sleep(5)

    print("taking right photo")
    rightImg = cam.photo()

    if saveImages:
        cv2.imwrite(imgPath + "Left.jpg", leftImg)
        cv2.imwrite(imgPath + "Right.jpg", rightImg)


leftImg = cv2.resize(leftImg, downScaleSize, interpolation = cv2.INTER_AREA)
rightImg = cv2.resize(rightImg, downScaleSize, interpolation = cv2.INTER_AREA)


def getDistance(left, right):
    # left = cv2.fastNlMeansDenoisingColored(left, None, 10, 10, 7, 15)
    # right = cv2.fastNlMeansDenoisingColored(right, None, 10, 10, 7, 15)

    left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoSGBM_create(numDisparities=128, blockSize=17)

    stereo.setTextureThreshold(10)

    disparity = stereo.compute(right, left)
    
    disparity = disparity / disparity.max() * 255

    cv2.imwrite(imgPath + "disparity.jpg", disparity)

    return disparity


getDistance(rightImg, leftImg)
