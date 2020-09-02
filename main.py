#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 22:37:27 2020

@author: aguasharo
"""

from __future__ import print_function

import json
import os
import itertools
import pandas as pd



from sklearn.manifold import TSNE
import seaborn as sns


import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.optimizers import Adam, SGD


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


from sklearn.preprocessing import StandardScaler

import collections
from collections import Counter

import multiprocessing as mp
from joblib import Parallel, delayed



from readDataset import *
from preProcessing import *
from featureExtraction import *
from classificationEMG import *




#%% Functions

from RecognitionModel import RecognitionModel
        




#%% Read user data
response = collections.defaultdict(dict)
num_gestures = 6
folderData = 'trainingJSON'

entries = os.listdir(folderData)




for entry in entries:
    file_selected = folderData + '/' + entry + '/' + entry + '.json'
    
    with open(file_selected) as file:
        
        # Read user data
        user = json.load(file)      
        name_user = user['userInfo']['name']
        print(name_user)  

        currentUser = RecognitionModel('training', user)     
        # Preprocessing
        train_segment_X  = currentUser.preProcessingData()
        
        # Feature extraction by computing the DTW distance between each training
        # example and the center of each cluster  
        [X_train, centers] = currentUser.featureExtraction(train_segment_X)
        
        # Training the feed-forward NN
        estimator = currentUser.trainSoftmaxNN(X_train)
        
        results = currentUser.classifyGestures('testing', estimator, centers)    
     
     # Concatenating the predictions of all the users for computing the
     # errors    
    response[name_user]['testing'] = results

           
with open('responses5.json', 'w') as json_file:
  json.dump(response, json_file)             

































             





                


            

        

           




            



