# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:04:33 2020

@author: Shaun Zacharia
"""

import pandas as pd

def scriptGen(folder):
    """
    Parameters
    ----------
    folder : str
        Name of folder in which the files "Block Data.csv" and "connections.csv"
        are present.

    Returns
    -------
    Creates the required MATLAB script to make the model.

    """    
    df = pd.read_csv(folder + "Block Data.csv")
    f = open(folder + "model_gen.m", 'w')
    # Initial statements required to create the model
    f.write("sys = 'testModel';\n")
    f.write("new_system(sys)\nload_system('simulink')\nopen_system(sys)\n")
    
    # placing all the blocks in the right position
    for index, row in df.iterrows():
        f.write("pos = [{} {} {} {}];\n".format(row['X'], row['Y'], 
                                               row['X'] + 45, row['Y'] + 45))
        f.write("add_block('{}', [sys '/{}'], 'Position', pos)\n".format(row['Block'], index))
    
    # reading the connections
    conn = pd.read_csv(folder + "connections.csv")
    
    # making all the connections, as the number of connections to each block 
    # increases, the port dictionarycontains the number of connections made to 
    # a particular block so far
    ports = {}
    for index, row in conn.iterrows():
        if row['To'] not in ports.keys():
            # adding block to list if not present
            ports[row['To']] = 1
            f.write("add_line(sys, '{}/1', '{}/1', 'autorouting', 'on')\n".format(row['From'], row['To']))
        else:
            # if already present, connect input to the next input port of the block
            ports[row['To']] += 1
            f.write("add_line(sys, '{}/1', '{}/{}', 'autorouting', 'on')\n".format(row['From'], row['To'], ports[row['To']]))
    f.close()