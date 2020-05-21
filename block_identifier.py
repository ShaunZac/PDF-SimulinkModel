# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:26:16 2020

@author: Shaun Zacharia
"""


import numpy as np
import pandas as pd
from keras import losses
from keras.models import load_model
import keras.backend as K
K.set_image_data_format('channels_last')

def identifyBlocks(folder, model):
    """
    Parameters
    ----------
    folder : str
        The name of the folder in which the Block Data and the regions data 
        is present.
    model : str
        File path of the model trained by CNN_train.

    Returns
    -------
    df : Pandas DataFrame
        Updates 'Block Data.csv' to now contain the name of ech corresponding
        block along with its location.

    """
    parts = ["simulink/Discrete/Unit Delay", 
             "simulink/Math Operations/Gain", 
             "simulink/Logic and Bit Operations/Relational Operator",
             "simulink/Sources/Step",
             "simulink/Sources/Sine Wave",
             "simulink/Sinks/Stop Simulation"]
    
    model = load_model(model)
    
    model.compile(loss=losses.categorical_crossentropy, optimizer='Adam', 
                  metrics=['accuracy'])
    
    df = pd.read_csv(folder + "Block Data.csv")
    regions = np.load(folder + "regions.npy")
    out = model.predict(regions/255.)
    classified = np.argmax(out, axis = 1)
    
    df["Block"] = [parts[i] for i in classified]
    df.to_csv(folder + "Block Data.csv", index = False)
    return df
