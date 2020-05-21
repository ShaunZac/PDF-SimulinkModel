# -*- coding: utf-8 -*-
"""
Created on Thu May 21 11:12:09 2020

@author: Shaun Zacharia
"""


import cv2
import pandas as pd
from pdf_jpg import convert_pdf
from segmentation import fill, removeLines, getEnclosed, getLinesArrows
from segmentation import getConnections, debugVerify
from block_identifier import identifyBlocks
from script_gen import scriptGen

# the path containing the pdf
file_path = r"C:\Users\Shaun Zacharia\Desktop\Airbus\trial stuff\enclosed_5.pdf"
# the folder in which the folder containing images is to be made
output_path = r"C:\Users\Shaun Zacharia\Documents\GitHub\PDF-SimulinkModel"


convert_pdf(file_path, output_path)


# name of the folder
folder = "enclosed_5/"
# reading the image from a particular image
img = cv2.imread(folder + "0.jpg", 1)


# convert the image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# convert the grayscale image to binary image
ret,thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
# fill the enclosed spaces
filled = fill(thresh, folder, save = True)
# remove the lines from the image
no_line = removeLines(filled, folder,save = True)
# get details of the enclosed spaces
coords, netlist, regions = getEnclosed(no_line, thresh, folder, save = True)
# isolate the lines and arrows
arrow_coords, lines = getLinesArrows(netlist, folder, save = True)
# get the connections of the blocks
conn = getConnections(lines, coords, arrow_coords, folder, save = True)
# only to be used if there is a fault in the model (visual checking)
debugVerify(lines, coords, arrow_coords, folder, save = True)
# saving the block data
block_data = pd.DataFrame(coords, columns = ['X', 'Y'])
block_data.to_csv(folder + "Block Data.csv", index = False)


# updating block data with the names of the blocks
block_data = identifyBlocks(folder, "model.h5")


# writing the MATLAB script
scriptGen(folder)
