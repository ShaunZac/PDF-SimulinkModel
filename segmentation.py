# -*- coding: utf-8 -*-
"""
Created on Thu May  7 08:45:18 2020

@author: Shaun Zacharia
"""


import cv2
import numpy as np

# reading the image
img = cv2.imread('enclosed/0.jpg', 1)
 
# convert the image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
# convert the grayscale image to binary image
ret,thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

def fill(image, save = False):
    """
    Parameters
    ----------
    image : Takes in a binary thresholded image
        Will also work with colour images, but the results won't be as good.
        
    save : Boolean
        If True, it will save the image in jpg format with the name 'filled.jpg'
        in the location of the working directory

    Returns
    -------
    mask : The array version of the filled image
    Creates image with all the enclosed spaces filled
    """
    # filling all enclosed areas
    h, w = image.shape
    seed = (w//2,h//2)
    mask = np.zeros((h+2,w+2),np.uint8)
    floodflags = 4
    floodflags |= cv2.FLOODFILL_MASK_ONLY
    floodflags |= (255 << 8)
    num,thresh,mask,rect = cv2.floodFill(image, mask, seed, (255,0,0), (10,)*3,
                                         (10,)*3, floodflags)
    
    # saving the mask
    if save:
        cv2.imwrite("filled.jpg", mask)
    return mask

filled = fill(thresh)