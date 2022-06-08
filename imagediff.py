from tkinter import BaseWidget
from PIL import Image
import numpy as np

img1 = np.asarray(Image.open("Imagenes/img950.jpg").resize((400, 400)))


img1 = img1[:, :, 2] # Convert to red channel only
img1[img1<200] = 0 # Remove anything that isn't a cloud

img2 = np.asarray(Image.open("Imagenes/img960.jpg").resize((400, 400)))


img2 = img2[:, :, 2]
img2[img2<200] = 0

diff = 0

print(img1.shape)

chunkSize = 64

Image.fromarray(img1).show()

for x in range(0, img1.shape[0]-chunkSize, chunkSize):
    for y in range(0, img1.shape[1]-chunkSize, chunkSize):
        slice1 = img1[x:x+chunkSize, y:y+chunkSize]# - img2[x:x+8, y:y+8]
        sum = np.sum(slice1)
        if sum > 0:
            min = 3000000000
            coords = [0, 0]
            for xx in range(img2.shape[0] - chunkSize):
                for yy in range(img2.shape[1]- chunkSize):


                    diff = np.abs(slice1 - img2[xx:xx+chunkSize, yy:yy+chunkSize])
                    if np.sum(diff) < min:
                        min = np.sum(diff)
                        coords[0] = xx
                        coords[1] = yy

            Image.fromarray(slice1).show()
            Image.fromarray(img2[coords[0]:coords[0]+chunkSize, coords[1]:coords[1]+chunkSize]).show()

            print(min, coords)
        