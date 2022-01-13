#
#  camera.py
#
#      A small utility to take photos
#
#
#  Required libraries:
#
#      # numpy
#      # picamera
#
#  Methods:
#
#      # photo() -> Take a photo
#



# Import necessary libraries

import numpy as np # To save image data on a numpy array
import picamera # To take the photos


# Initialize objects

camera = picamera.PiCamera() # PiCamera object


# Set camera configuration

camera.resolution = (2592, 1944) # We set resolution to 2592 by 1944
camera.framerate = 1 # Since we are only taking photos, not videos, we don't need more than that


# Methods

def photo(): # A method to take a photo and return it as a numpy array
    output = np.empty((1952, 2592, 3), dtype=np.uint8) # Create an empty numpy array called "output"
    camera.capture(output, 'rgb') # Use picamera to capture the actual photo
    output = np.flip(output, axis=2) # Use np.flip() method to invert the array in the third axis, the color (changes RGB format to BGR)
    output = np.rot90(output, k=2) # Use np.rot90() method to rotate the image 180 degrees (k=2 means we want to rotate 180deg instead of 90deg)
    return output # Return the image