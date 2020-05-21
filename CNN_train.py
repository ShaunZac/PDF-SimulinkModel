# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:50:48 2020

@author: Shaun Zacharia
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from keras import losses
from keras import initializers
from keras.layers import Input, Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Model
import keras.backend as K
K.set_image_data_format('channels_last')

def initializeData():
    """
    Returns
    -------
    X_train : ndarray
        Train input, shape = (no. of train examples, output_shape, output_shape, 1).
    y_train : ndarray
        Train output, shape = (no. of train examples, no. of blocks).
    X_test : ndarray
        Test input, shape = (no. of test examples, output_shape, output_shape, 1).
    y_test : ndarray
        Test output, shape = (no. of test examples, no. of blocks).
    """
    # reading and reshaping the data for desired form of keras model
    X_train = cv2.imread("data/delay (1).jpg", 0)
    X_train = X_train.reshape((1, 120, 120, 1))
    parts = ["delay", "gain", "gre", "ramp", "sine", "stop"]
    for name in parts:
        for i in range(1, 16):
            if(name=='delay' and i==1):
                continue
            path = "data/"+name+" ({}).jpg".format(i)
            im = cv2.imread(path, 0)
            im = im.reshape((1, 120, 120, 1))
            X_train = np.concatenate((X_train, im), axis = 0)
            
    X_test = cv2.imread("data/delay (16).jpg", 0)
    X_test = X_test.reshape((1, 120, 120, 1))
    for name in parts:
        for i in range(16, 20):
            if(name=='delay' and i==16):
                continue
            path = "data/"+name+" ({}).jpg".format(i)
            im = cv2.imread(path, 0)
            im = im.reshape((1, 120, 120, 1))
            X_test = np.concatenate((X_test, im), axis = 0)
    
    # normaizing data (this is sufficient for image data)
    X_train = X_train/255.
    X_test = X_test/255.
    
    y_train = sorted(parts*15)
    y_test = sorted(parts*4)
    
    # one-hot encoding the y data
    y_train = pd.get_dummies(pd.Series(y_train).astype('category')).to_numpy()
    y_test = pd.get_dummies(pd.Series(y_test).astype('category')).to_numpy()
    
    return X_train, y_train, X_test, y_test

def myModel(input_shape, seed = 0):
    """
    Parameters
    ----------
    input_shape : tuple
        The shape of the input in form (output_size, output_size, 1).
    seed : int, optional
        Define the seed to get reproducible results. The default is 0.

    Returns
    -------
    model : keras Model
        This Keras model can be used to train on the input images.

    """
      
    # Define the input placeholder as a tensor with shape input_shape
    X_input = Input(input_shape)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(8, (7, 7), padding = 'same', 
               kernel_initializer = initializers.he_uniform(seed=seed))(X_input)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2))(X)
    
    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(16, (7, 7), padding = 'same',
               kernel_initializer = initializers.he_uniform(seed=seed))(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2))(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(6, activation='sigmoid',
              kernel_initializer = initializers.he_uniform(seed=seed))(X)

    # Create model
    model = Model(inputs = X_input, outputs = X, name='myModel')    
    
    return model


def modelPerformance(history):
    """
    Parameters
    ----------
    history : keras History
        The history returned while fitting the model.

    Returns
    -------
    The plots of how accuracy and loss for train and test data vary with each 
    epoch and the final accuracy and loss of the model on the test data.

    """
    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title("Loss")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.title("Accuracy")
    plt.legend()
    plt.show()
    
    preds = model.evaluate(X_test, y_test)
    print()
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))


X_train, y_train, X_test, y_test = initializeData()

model = myModel(X_train.shape[1:])

# compile and fit the model
model.compile(loss=losses.categorical_crossentropy, optimizer='Adam', 
              metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data = (X_test, y_test), 
                    epochs = 7, batch_size = 16)
model.save('model.h5')