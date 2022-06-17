from tkinter import BaseWidget
from PIL import Image
import numpy as np
import numba as nb
import time


cloudThreshold = 200
chunkSize = 40
differenceThreshold = 0
imageRes = 400


img1 = np.asarray(Image.open("Imagenes/test1.png").resize((imageRes, imageRes)), dtype=np.uint8).copy() # load image, resize, convert to uint8, and copy

img1 = img1[:, :, 2] # Convert to red channel only
img1[img1<cloudThreshold] = 0 # convert to binary
img1[img1>0] = 1


img2 = np.asarray(Image.open("Imagenes/test2.png").resize((imageRes, imageRes)), dtype=np.uint8).copy()

img2 = img2[:, :, 2]
img2[img2<cloudThreshold] = 0
img2[img2>0] = 1


Image.fromarray(img1 * 150).show()



@nb.jit(nopython=True, parallel=True)
def findImage(img1, img2, chunkSize, UsableThreshold):
    outSizeX = np.uint8(img1.shape[0]/chunkSize) # calculate the number of chunks
    outSizeY = np.uint8(img1.shape[1]/chunkSize)

    coords = np.zeros((outSizeX, outSizeY, 2), dtype=np.uint16) # create an array to store all the coordinates of the found image

    for x in nb.prange(outSizeX): # use a parralel loop
        for y in nb.prange(outSizeY):
            slice1 = img1[x*chunkSize:(x+1)*chunkSize, y*chunkSize:(y+1)*chunkSize] # multiply the current chunk by the chunksize to get the image slice



            if np.sum(slice1) > UsableThreshold: # if there are more than 200 pixels
                min = 10000000

                for xx in range(img2.shape[0] - chunkSize): # loop over every pixel of the second image
                    for yy in range(img2.shape[1] - chunkSize):
                        slice2 = img2[xx:xx+chunkSize, yy:yy+chunkSize] # get a slice of the second image
                        difference = np.sum(slice1^slice2) # calculate the sum of the xor, the lower it is, the more similar the images are

                        if difference <= min:
                            min = difference # update the minimum
                            coords[x, y] = [xx, yy] # and store the coordinates

                # print(min,"sum", np.sum(slice1), x, y, " ", coords[x][y])
                # sample1 = img1.copy()
                # sample1[x*chunkSize:(x+1)*chunkSize, y*chunkSize:(y+1)*chunkSize] *= 150
                #
                # sample2 = img2.copy()
                # sample2[coords[x][y][0]:coords[x][y][0]+chunkSize, coords[x][y][1]:coords[x][y][1]+chunkSize] *= 150
                # Image.fromarray(sample1).show()
                # Image.fromarray(sample2).show()

    return coords


# first time is slower
startTime = time.time()
print(findImage(img1, img2, chunkSize, 100))
print("took", time.time() - startTime, "seconds")

startTime = time.time()
coords = findImage(img1, img2, chunkSize, 100)
print("took", time.time() - startTime, "seconds")

# show results
for x in range(coords.shape[0]):
    for y in range(coords.shape[1]):
        if not (coords[x][y][0] == 0):
            sample1 = img1.copy()
            sample1[x*chunkSize:(x+1)*chunkSize, y*chunkSize:(y+1)*chunkSize] *= 150

            sample2 = img2.copy()
            sample2[coords[x][y][0]:coords[x][y][0]+chunkSize, coords[x][y][1]:coords[x][y][1]+chunkSize] *= 150
            Image.fromarray(sample1).show()
            Image.fromarray(sample2).show()
