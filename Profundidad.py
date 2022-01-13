#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import cv2 as cv


# In[3]:


# get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt


# In[4]:


imgL = cv.imread('st1l.jpg',cv.IMREAD_GRAYSCALE)
# plt.imshow(imgL,'gray')


# In[5]:


imgR = cv.imread('st1r.jpg',cv.IMREAD_GRAYSCALE)
# plt.imshow(imgR,'gray')

downScaleSize = (1500, 1000)

imgL = cv.resize(imgL, downScaleSize, interpolation = cv.INTER_AREA)
imgR = cv.resize(imgR, downScaleSize, interpolation = cv.INTER_AREA)

# In[6]:
print(imgR.shape)

stereo = cv.StereoBM_create(numDisparities=400, blockSize=29)


# In[7]:


print(stereo.getPreFilterCap())
print(stereo.getPreFilterSize())
print(stereo.getPreFilterType())
print(stereo.getROI1())
print(stereo.getROI2())
print(stereo.getSmallerBlockSize())
print(stereo.getTextureThreshold())
print(stereo.getUniquenessRatio())


# In[8]:


# stereo.setPreFilterCap(31)
stereo.setTextureThreshold(500)
stereo.setUniquenessRatio(11)
# stereo.setSmallerBlockSize(1)
# stereo.setPreFilterSize(9)

# In[9]:


print(stereo.getPreFilterCap())
print(stereo.getPreFilterSize())
print(stereo.getPreFilterType())
print(stereo.getROI1())
print(stereo.getROI2())
print(stereo.getSmallerBlockSize())
print(stereo.getTextureThreshold())
print(stereo.getUniquenessRatio())


# In[10]:


disparity = stereo.compute(imgR,imgL)


# In[11]:


font = cv.FONT_HERSHEY_SIMPLEX
parametros = ["PreFilterCap: {}".format(stereo.getPreFilterCap()),
              "PreFilterSize: {}".format(stereo.getPreFilterSize()),
             "PreFilterType: {}".format(stereo.getPreFilterType()),
             "SmallerBlockSize: {}".format(stereo.getSmallerBlockSize()),
             "TextureThreshold: {}".format(stereo.getTextureThreshold()),
             "UniquenessRatio: {}".format(stereo.getUniquenessRatio())]
              
posInicial = (30,2500)
for param in parametros:
    cv.putText(disparity, param, posInicial, font, 2, (255, 255, 255), 3, cv.LINE_AA)
    posInicial = (posInicial[0], posInicial[1]+55)


# In[12]:


# plt.imshow(disparity,'gray')


# In[ ]:





# In[13]:


plt.show()


# In[14]:

print(disparity.max())
disparity = disparity / disparity.max() * 255
print(disparity.max())
cv.imwrite('output.jpg', disparity)

