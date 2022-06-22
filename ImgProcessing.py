from PIL import Image, ImageFilter
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

def loadAndProcessImage(fileName, cloudThreshold, imageRes):
    img = np.asarray(Image.open(fileName).crop((400, 80, 2300, 1950)).resize((imageRes, imageRes)), dtype=np.uint8).copy() # load image, resize, convert to uint8, and copy

    img = img[:, :, 2] # Convert to red channel only
    img[img<cloudThreshold] = 0 # convert to binary
    img[img>0] = 1

    return img

# img1, img2: images used to find in one another
# chunkSize: size of the image slices used for comparing. higher = less resolution, more accuracy. lower = more resolution, less accuracy
# cloudPercentageMin, cloudPercentageMax: used to see if a chunk is suitable for comparison. between 0 and 1
@nb.njit(parallel=True) # compile with numba and run in parallel
def compareImages(img1, img2, chunkSize, cloudPercentageMin, cloudPercentageMax):

    minPixels = chunkSize * chunkSize * cloudPercentageMin # calculate the amount of pixels from the percantage
    maxPixels = chunkSize * chunkSize * cloudPercentageMax

    outSizeX = np.uint8(img1.shape[0]/chunkSize) # calculate the number of chunks
    outSizeY = np.uint8(img1.shape[1]/chunkSize)

    coords = np.zeros((outSizeX, outSizeY, 2), dtype=np.uint16) # create an array to store all the coordinates of the found image

    for x in nb.prange(outSizeX): # use a parralel loop
        for y in nb.prange(outSizeY):
            slice1 = img1[x*chunkSize:(x+1)*chunkSize, y*chunkSize:(y+1)*chunkSize] # multiply the current chunk by the chunksize to get the image slice

            sum = np.sum(slice1) # sum of the pixels

            if minPixels < sum and sum < maxPixels: # if the amount of pixels is in range
                min = 10000000

                for xx in nb.prange(img2.shape[0] - chunkSize): # loop over every pixel of the second image
                    for yy in nb.prange(img2.shape[1] - chunkSize):
                        difference = np.sum(slice1^img2[xx:xx+chunkSize, yy:yy+chunkSize]) # calculate the sum of the xor, the lower it is, the more similar the images are

                        if difference <= min:
                            min = difference # update the minimum
                            coords[x, y] = [xx, yy] # and store the coordinates

            if sum > maxPixels:
                coords[x, y] = [65535, 65535] # mark a mostly white area, to later refill it

    return coords

# coords: previously calculated coordinates
# chunkSize: chunk size used to calculate the coordinates
# distanceThreshold: max amount of distance acceptable
@nb.njit
def calculateDistance(coords, chunkSize, distanceThreshold):
    distances = np.zeros((coords.shape[0], coords.shape[1])) # create array to store the distances
    lastDistance = 0
    for x in range(coords.shape[0]): # loop over the coordinates
        for y in range(coords.shape[1]):
            if not (coords[x][y][0] == 0 and coords[x][y][1] == 0) : # if the coordinates are at 0, 0, dont include
                if (coords[x][y][0] == 65535 and coords[x][y][1] == 65535):
                    distances[x, y] = np.max(np.asarray([distances[x-1, y], distances[x+1, y], distances[x, y-1], distances[x, y+1]]))

                else:
                    distance = np.sqrt(np.power(coords[x][y][0] - x*chunkSize, 2) + np.power(coords[x][y][1] - y*chunkSize, 2)) # calculate how many pixels away

                    if distance < distanceThreshold and distance != 0: # if the distance is acceptable
                        distances[x, y] = distance # update distance
                        lastDistance = distance # update last distance

    return distances

# visualize the output
def visualizeData(distances, img, distanceThreshold):
    processedDistances = Image.fromarray(distances / distanceThreshold * 255).resize((img.shape[0], img.shape[1])) # convert to image and resize
    processedDistances = processedDistances.convert('L').filter( ImageFilter.GaussianBlur(radius=25) ) # convert to L datatype and blur

    processedDistances = np.asarray(processedDistances) / 255 # scale from 0 to 1

    processedDistances[img==0] = 0 # mask using the original image

    plt.imshow(processedDistances, cmap='gnuplot2') # cmap is the colors used
    plt.show()
