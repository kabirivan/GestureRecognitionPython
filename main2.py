#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 23:50:02 2020

@author: aguasharo
"""


import json
import os
import itertools
import pandas as pd
import math
from scipy import signal
import numpy as np
import simplespectral as sp
import matplotlib.pyplot as plt
import imagesc as imagesc

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


folderData = 'trainingJSON'
hand_gestures = ['noGesture', 'fist', 'waveIn', 'waveOut', 'open', 'pinch']


files = []

for root, dirs, files in os.walk(folderData):
     print('Dataset Ready !')
         
         
file_selected = root + '/' + files[2]  

with open(file_selected) as file:
    user = json.load(file)     

# Training Process
train_samples = user['trainingSamples']



for move in hand_gestures:
    for i in range(1,26):        
        train_sample = train_samples[move]['sample%s' %i]['emg']
    print(move)    
    
    
