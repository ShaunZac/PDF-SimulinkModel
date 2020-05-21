# -*- coding: utf-8 -*-
"""
Created on Wed May  6 09:17:55 2020

@author: Shaun Zacharia
"""

import os
import tempfile
from pdf2image import convert_from_path
# Run "pip install pdf2image" to install the required dependencies

def convert_pdf(file_path, output_path = ''):
    """
    Parameters
    ----------
    file_path : str,
        The path of the pdf you want to convert. E.g. 'C:\example.pdf'
    output_path : str, 
        The output directory in which you want the . 
        The default is '', i.e the same as the working directory.

    Returns
    -------
    None.
    Converts the PDF to .jpg format with each page saved in a folder with the
    name same as the pdf's name.

    """
    # save temp image files in temp dir so that extra files created do not clutter our folder
    with tempfile.TemporaryDirectory() as temp_dir:
        # convert pdf to multiple image
        images = convert_from_path(file_path, output_folder=temp_dir)
        # Getting the absolute path of the output directory
        output_path = os.path.abspath(output_path)
        # Creating a folder in the output path
        path = os.path.join(output_path, os.path.basename(file_path)[:-4])
        os.mkdir(path)
        # save images
        for i in range(len(images)):
            file_path = str(i) + '.jpg'
            image_path = os.path.join(path, file_path)
            images[i].save(image_path, 'JPEG')
    return