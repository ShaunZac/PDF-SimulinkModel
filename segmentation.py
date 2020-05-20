# -*- coding: utf-8 -*-
"""
Created on Thu May  7 08:45:18 2020

@author: Shaun Zacharia
"""


import cv2
import numpy as np
import pandas as pd

folder = "enclosed_3/"
# reading the image
img = cv2.imread(folder + "0.jpg", 1)
 
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
        cv2.imwrite(folder + "filled.jpg", mask)
    return mask

filled = fill(thresh, save = True)

def removeLines(image, itr = 3, save = False):
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
    no_line = cv2.dilate(image, kernel, iterations=itr)
    if save:
        cv2.imwrite(folder + "no lines.jpg", no_line)
    return no_line

no_line = removeLines(filled, save = True)

def getEnclosed(no_line, image, output_size = 120, save = False):
    """
    Parameters
    ----------
    no_line : ndarray
        Takes the image with eroded lines as input, has to be binary image.
    image : ndarray
        Takes the image from which segmented parts should be cropped out.
        Will work best if binary thresholded image is given as argument.
    output_size : int, optional
        All the cropped images will be padded to become of this size, 
        its value should be greater than the width/height of each Region 
        of Interest. The default is 120.
    save : Bool, optional
        If true, it saves each region of interest in the working directory
        with the nameing convention 'ROI_i.jpg', where i is the i'th image.
        The default is False.

    Returns
    -------
    centroids : ndarray
        Contains a list of tuples, converted to array, where the ith element 
        corresponds to (x, y) coordinates of the ith ROI. shape = (no. of points, 2)
    original : ndarray
        Contains the netlist in image format
    regions : ndarray of shape (num_regions, output_size, output_size)
        Contains each region of interest in array format.

    """
    # n is amount by which to widen area since we get eroded image
    # 15 = (size of dilating kernel)*(num of iterations) = 5*3
    n = 15
    ROI_number = 0
    original = image.copy()
    
    # getting contours in the image, 255-no_line so that base is black
    cnts = cv2.findContours(255-no_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    # creating empty stack in which all segmented images will be stored
    regions = np.zeros((len(cnts), output_size, output_size))
    
    # creating an empty list to keep track of centers of blocks
    centroids = []
    
    for i, c in enumerate(cnts):
        x,y,w,h = cv2.boundingRect(c)
        # appending centers of blocks in the list
        centroids.append((x+w//2, y+h//2))
        ROI = image[y-n : y+h+n, x-n+5 : x+w+n+5]
        
        # setting to 0 so that only netlist can be seen
        original[y-n : y+h+n, x-n+5 : x+w+n+5] = 0
        
        ww = hh = output_size
        ht, wd = ROI.shape
        # compute center offset
        xx = (ww - wd) // 2
        yy = (hh - ht) // 2
        regions[i, yy:yy+ht, xx:xx+wd] = ROI
        if save:
            cv2.imwrite(folder + "ROI_{}.jpg".format(ROI_number), regions[i, :, :])
        ROI_number += 1
    if save:
        cv2.imwrite(folder + "netlist.jpg", original)
    return np.array(centroids), original, regions
        
coords, netlist, regions = getEnclosed(no_line, thresh, save = True)

def getLinesArrows(netlist, save = False):
    """
    Parameters
    ----------
    netlist : ndarray
        Should contain netlist with arrows.
    save : bool, optional
        If True, it saves the arrows and lines images. The default is False.

    Returns
    -------
    arrow_coords : ndarray
        Contains list of tuples, converted to array, where the ith element corresponds to 
        (x, y) coordinates of centroid of arrows. shape = (no. of points, 2)
    lines : ndarray
        Contains the array of the image containing just the lines.
    """
    
    kernel = np.ones((3, 3), np.uint8)
    # n is amount by which to widen area since we get eroded image
    n = 3
    arrow = cv2.erode(netlist, kernel, iterations=1)
    
    if save:
        cv2.imwrite(folder + "arrows.jpg", arrow)
    
    # getting the contours of the arrow image
    cnts, _ = cv2.findContours(arrow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = netlist.copy()
    arrow_coords = []
    
    for cnt in cnts:
        # checking is the arrow is a circle (if circle, it is left in the lines image)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt,True)
        pi = 3.14159265
        circularity = 4*pi*(area/perimeter**2)
        
        # setting a threshold for circularity
        if circularity < 0.75:
            x,y,w,h = cv2.boundingRect(cnt)
            arrow_coords.append((x+w//2, y+h//2))
            # setting the extracted part to 0 so that arrows are not in 'lines'
            lines[y-n : y+h+n, x-n : x+w+n] = 0
    if save:
        cv2.imwrite(folder + "lines.jpg", lines)
    
    return np.array(arrow_coords), lines

arrow_coords, lines = getLinesArrows(netlist, save = True)

def closestPoint(left, right):
    """
    Parameters
    ----------
    left : ndarray
        Takes array of points of shape (no of points, 2), this should contain
        either the coordinates of the arrows, or the starting points of lines.
    right : ndarray
        Takes array of points of shape (no of points, 2), this should always
        contain the coordinates of the boxes (coords).

    Returns
    -------
    corresp_box : list
        Contains the serial number of the box which is nearest to each point in 
        the array named 'left'.

    """
    corresp_box = []
    for node in left:
        distsq = np.sum((right - node)**2, axis = 1)
        corresp_box.append(distsq.argmin())
    return corresp_box

def getConnections(lines, coords, arrow_coords, save = False):
    """
    Parameters
    ----------
    lines : ndarray
        The image containing only the lines.
    coords : ndarray
        The coordinates of the boxes.
    arrow_coords : ndarray
        The coordinates of the arrows.
    save : Bool, optional
        Saves the connections in a CSV format. The default is False.

    Returns
    -------
    conn : pandas DataFrame
        Contains the connection data, which block number is connected to 
        which block number is given.

    """
    # Finding the contours of the lines
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # defining 2 kernels, one for line with 1 pixel width, one for 2 pixel width
    # these will help in detecting the starting points of the connections
    kernel = np.array(
        [[-1, -1, -1],
         [-1, 1, 0],
         [-1, -1, -1]], np.int32)
    kernel2 = np.array(
        [[0, -1, -1, -1, 0],
         [-1, -1, 1, 0, 0],
         [-1, -1, 1, 0, 0],
         [0, -1, -1, -1, 0],
         [0, 0, 0, 0, 0]])
    
    # the offset by which the bounding box is to be made larger to get the 
    # arrows in the bounding boxes
    n = 15
    conn = []
    
    # finding which arrow corresponds to which box
    arrow_blocks = closestPoint(arrow_coords, coords)
    
    # singling out each contour (line) and finding out its starting point and 
    # corresponding arrow via hit-miss morphological transform and checking which
    # arrow comes in which bounding box
    for i, cnt in enumerate(contours):
        skel = np.zeros((1700, 2200), np.uint8)
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.drawContours(skel, contours, i, 255, 1)
        
        # one of endpts, endpts2 has to be empty as a contour cannot have both
        endpts = np.column_stack(np.where(cv2.morphologyEx(skel, 
                                                            cv2.MORPH_HITMISS, kernel)>0))
        endpts2 = np.column_stack(np.where(cv2.morphologyEx(skel, 
                                                            cv2.MORPH_HITMISS, kernel2)>0))
        # ignoring the one which is empty
        if len(endpts)==0:
            start_coords = endpts2
        else:
            start_coords = endpts
            
        # reversing the order, so that it is compatible with openCV format
        start_coords = start_coords[:,[1, 0]]
        
        # checking which arrows lie inside the bounding box
        blocks = np.where(((x-n < arrow_coords[:, 0]) & (arrow_coords[:, 0] < x + w + n)) 
                          & ((y-n < arrow_coords[:, 1]) & (arrow_coords[:, 1] < y + h + n)))
        
        # making a list called connections, that will hold the connection data
        for j in blocks[0]:
            conn.append((closestPoint(start_coords, coords)[0], arrow_blocks[j]))
    
    # saving it in csv format
    conn = pd.DataFrame(np.array(conn), columns = ['From', 'To']).drop_duplicates()
    if save:
        conn.to_csv(folder + "connections.csv", index = False)
    
    return conn

conn = getConnections(lines, coords, arrow_coords, save = True)

def debugVerify(lines, save = False):    
    """
    Parameters
    ----------
    lines : ndarray
        The image containing only the lines.
    save : Bool, optional
        Saves the image that can be used to verify if the connections given by 
        getConnections is correct. The default is False.

    Returns
    -------
    trial : ndarray
        Contains every line individually circled, with the box number nearest 
        to each endpoint marked in the image.

    """
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    n = 15
    trial = np.zeros((1700, 2200, 3), np.uint8)
    for i in range(3):
        trial[:,:,i] = lines.copy()
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(trial,(x-n,y-n),(x+w+n,y+h+n),(0,255,0),2)
    
    skel = np.zeros((1700, 2200), np.uint8)
    cv2.drawContours(skel, contours, -1, 255, 1)
    cv2.imwrite(folder + "skeleton.jpg", skel)
    
    kernel = np.array(
        [[-1, -1, -1],
          [-1, 1, 0],
          [-1, -1, -1]], np.int32)
    endpts = np.column_stack(np.where(cv2.morphologyEx(skel, cv2.MORPH_HITMISS, kernel)>0))
    
    kernel2 = np.array(
        [[0, -1, -1, -1, 0],
          [-1, -1, 1, 0, 0],
          [-1, -1, 1, 0, 0],
          [0, -1, -1, -1, 0]])
    endpts2 = np.column_stack(np.where(cv2.morphologyEx(skel, cv2.MORPH_HITMISS, kernel2)>0))
    
    start_coords = np.concatenate((endpts, endpts2), axis = 0)
    start_coords = start_coords[:,[1, 0]]
    
    
    start_blocks = closestPoint(start_coords, coords)
    arrow_blocks = closestPoint(arrow_coords, coords)
        
    for idx, i in enumerate(start_coords):
        cX, cY = i
        cv2.circle(trial, (cX, cY), 5, (0, 0, 255), -1)
        cv2.putText(trial, str(start_blocks[idx]), (cX - 10, cY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    for idx, i in enumerate(arrow_coords):
        cX, cY = i
        cv2.circle(trial, (cX, cY), 5, (255, 0, 0), -1)
        cv2.putText(trial, str(arrow_blocks[idx]), (cX - 10, cY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
    if save:
        cv2.imwrite(folder + "trial.jpg", trial)
    return trial

debugVerify(lines, save = True)