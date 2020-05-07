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
    h, w = image.shape[:2]
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

def removeLines(image, save = False):
    """
    Parameters
    ----------
    image : ndarray
        The input image from which lines are to be eroded.
    save : Boolean, optional
        If true, it saves the no line image in the working directory with the
        name 'no lines.jpg'. The default is False.

    Returns
    -------
    no_line : ndarray
        Performs morphological dilation to cause the lines to get eroded.
    """
    
    kernel = np.ones((5, 5), np.uint8)
    
    # morphological operation to remove the lines
    no_line = cv2.dilate(filled, kernel, iterations=3)
    if save:
        cv2.imwrite("no lines.jpg", no_line)
    return no_line

no_line = removeLines(filled, save = True)

def getEnclosed(no_line, image, save = False):
    
    # n is amount by which to widen area since we get eroded image
    # 11 = 5 + (5-2)*(no. of iterations)
    n = 11
    ROI_number = 0
    original = image.copy()
    
    # getting contours in the image, 255-no_line so that base is black
    cnts = cv2.findContours(255-no_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        #cv2.rectangle(image, (x-n, y-n), (x + w + n, y + h + n), (36,255,12), 2)
        ROI = original[y-n : y+h+n, x-n : x+w+n]
        if save:
            cv2.imwrite('enclosed\ROI_{}.jpg'.format(ROI_number), ROI)
        ROI_number += 1
    return
        
getEnclosed(no_line, thresh)
