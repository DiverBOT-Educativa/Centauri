from tkinter import BaseWidget
from PIL import Image, ImageFilter
import numpy as np
import numba as nb
import time
import matplotlib.pyplot as plt
from ImgProcessing import *
import random

cloudThreshold = 190 # brightness at which clouds are included
imageRes = 600 # resolution of the downscaled image


chunkSize = 20 # size of the chunks. higher = less resolution, more accuracy. lower = more resolution, less accuracy
minCloudPercentage = 0.1 # used to determine if a chunk is suitable to check
maxCloudPercentage = 0.95

distanceThreshold = 50 # max distance between two points that is acceptable



while True:
    absoluteStartTime = time.time()
    print("loading and processing images...")
    randomImg = random.randrange(0, 1882) # select a random image
    print("image index:", randomImg)
    img1 = loadAndProcessImage("img/img" + str(randomImg) + ".jpg", cloudThreshold, imageRes) # load and process both images
    img2 = loadAndProcessImage("img/img" + str(randomImg + 1) + ".jpg", cloudThreshold, imageRes)

    print("image loading and processing took " + str(time.time() - absoluteStartTime) + " seconds")
    print()
    Image.open("img/img" + str(randomImg) + ".jpg").show() # show images
    Image.fromarray(img1 * 150).show()

    startTime = time.time()
    print("comparing the image...")

    coords = compareImages(img1, img2, chunkSize, minCloudPercentage, maxCloudPercentage)

    print("image comparison took " + str(time.time() - startTime) + " seconds")
    print()
    startTime = time.time()
    print("calculating distances...")

    distances = calculateDistance(coords, chunkSize, distanceThreshold)

    print("calculating distances took " + str(time.time() - startTime) + " seconds")
    print()
    print("----------------------------------------------------------------")
    print("total loop time: " + str(time.time() - absoluteStartTime) + " seconds")
    print("----------------------------------------------------------------")
    print()
    visualizeData(distances, img1, distanceThreshold, randomImg) # visualize
