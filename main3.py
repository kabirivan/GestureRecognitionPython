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


        
def get_x_train(user,sample):
    # This function reads the time series(x) of the user (Training Sample)
    train_samples = user['trainingSamples']
    x = (train_samples[sample]['emg'])
    # Divide to 128 for having a signal between -1 and 1
    df = pd.DataFrame.from_dict(x) / 128
    # Apply filter
    train_filtered_X = df.apply(preProcessEMGSegment)
    # Segment the filtered EMG signal
    train_segment_X = EMG_segment(train_filtered_X)
    
    return train_segment_X


            
def get_x_test(user,sample):
    # This function reads the time series(x) of the user (Testing Sample)
    test_samples = user['testingSamples']
    x = (test_samples[sample]['emg'])
    df = pd.DataFrame.from_dict(x) / 128
    
    return df



def decode_targets(y_train):
    # Encode targets to train the Neural Network
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    target = np_utils.to_categorical(encoded_Y)
    
    return target 

    
def get_xy_val(X_train, targets):            
    
    # Get validation data to train Neural Network            
    data_val = X_train.copy()
    data_val['6'] = targets
    
    xy_val = data_val.sample(frac=1).reset_index(drop=True)
    
    
    X_val = xy_val.iloc[:,0:6]  
    y_val = decode_targets(xy_val['6'])
    
    
    return X_val, y_val



def bestCenter_Class(train_segment_X):
    
    # This function returns a set of time series called centers
    # for each gesture class
    
    g1 = train_segment_X[0:25]
    g2 = train_segment_X[25:50]
    g3 = train_segment_X[50:75]
    g4 = train_segment_X[75:100]
    g5 = train_segment_X[100:125]
    g6 = train_segment_X[125:150]
    
    gen = [g1, g2, g3, g4, g5 ,g6]
    
    c = [findCentersClass(g) for g in gen]
             
    return c



def getFeatureExtraction(emg_filtered, centers):
    
    features = featureExtractionf(emg_filtered, centers)  
    dataX = preProcessFeatureVector(features)
    
    return dataX
    
    

def trainFeedForwardNetwork(X_train,y_train, X_test, y_test):
    # This function trains an  artificial feed-forward neural networks 
    # Cost lost: categorical cross entropy
    # Hidden Layer: Tanh
    # Output Layer: softmax
    
    classifier = Sequential()
    
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = None, input_dim = 6))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'tanh'))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    classifier.fit(X_train, y_train, batch_size = 150, epochs = 1500, validation_data = (X_test, y_test), verbose = 0 )
    
    return classifier



# def classify_gesture(test_RawX, centers, estimator): 
#     # This function applies a hand gesture recognition model based on artificial
#     # feed-forward neural networks and automatic feature extraction to a set of
#     # EMGs conatined in the set test_RawX
    
#     [predictedSeq, vec_time, time_seq]= classifyEMG_SegmentationNN(test_RawX, centers, estimator)
#     predicted_label, t_post = post_ProcessLabels(predictedSeq)
    
#     # Computing the time of processing
#     estimatedTime =  [sum(x) for x in zip(time_seq, t_post)]
       
#     return predicted_label, predictedSeq, vec_time, estimatedTime



def testing_prediction(user,sample,centers,estimator): 
    test_RawX = get_x_test(user,sample) 
    [predictedSeq, vec_time, time_seq]= classifyEMG_SegmentationNN(test_RawX, centers, estimator)
    predicted_label, t_post = post_ProcessLabels(predictedSeq)
    
    # Computing the time of processing
    estimatedTime =  [sum(x) for x in zip(time_seq, t_post)]

    
    return predicted_label, predictedSeq, vec_time, estimatedTime



def recognition_results(results):  
     # This function save the responses of each user into a dictionary

    d = collections.defaultdict(dict)
    
    for i in range(0,150):
                
        d['idx_%s' %i]['class'] = code2gesture(results[i][0])
        d['idx_%s' %i]['vectorOfLabels'] = code2gesture_labels(results[i][1])
        d['idx_%s' %i]['vectorOfTimePoints'] = results[i][2]
        d['idx_%s' %i]['vectorOfProcessingTime']= results[i][3]    
       
    return d



#%% Read user data
response = collections.defaultdict(dict)
num_gestures = 6
folderData = 'trainingJSON'

entries = os.listdir(folderData)


class RecognitionModel:
    
    num_gestures = 6
    
    def __init__(self,version,user):
        self.user = user
        self.version = version
        
     
    def preProcessingData(self):
        sample_type = self.version+'Samples'
        # Reading the training samples
        train_samples = self.user[sample_type]
        # Preprocessing
        train_segment_X = [get_x_train(self.user,sample) for sample in train_samples] 
        
        return train_segment_X 
   
    def featureExtraction(self, train_data):         
        # Finding the EMG that is the center of each class
        centers = bestCenter_Class(train_data)  
        # Feature extraction by computing the DTW distance between each training
        # example and the center of each cluster           
        # Preprocessing the feature vectors    
        X_train = getFeatureExtraction(train_data, centers)
         
        return X_train, centers
         
         
    def trainSoftmaxNN(self, X_train):
        sample_type = self.version+'Samples'
        # Reading the training samples
        train_samples = self.user[sample_type]      
        # Training the feed-forward NN
        y_train = decode_targets(get_y_train(train_samples)) 
        X_val, y_val = get_xy_val(X_train, get_y_train(train_samples)) 
        estimator = trainFeedForwardNetwork(X_train, y_train, X_val, y_val)          
        
        return estimator
       
      
    def classifyGestures(self,version, estimator, centers) :
        
        sample_type = self.version+'Samples'
        # Reading the testing samples    
        test_samples = self.user[sample_type]      
        # Concatenating the predictions of all the users for computing the
        # errors
        results = ([testing_prediction(self.user, sample, centers, estimator) for sample in test_samples]) 
        
        return results
    


  


for entry in entries:
    file_selected = folderData + '/' + entry + '/' + entry + '.json'
    
    with open(file_selected) as file:
        user = json.load(file)      
        name_user = user['userInfo']['name']
        print(name_user)  

        currentUser = RecognitionModel('training', user)     
        
        train_segment_X  = currentUser.preProcessingData()
        
        [X_train, centers] = currentUser.featureExtraction(train_segment_X)
        
        estimator = currentUser.trainSoftmaxNN(X_train)
        
        results = currentUser.classifyGestures('testing', estimator, centers)    
        
    response[name_user]['testing'] = recognition_results(results)

           
with open('responses5.json', 'w') as json_file:
  json.dump(response, json_file)             





























# for entry in entries:
#     file_selected = folderData + '/' + entry + '/' + entry + '.json'
    
#     with open(file_selected) as file:
#         user = json.load(file)      
#         name_user = user['userInfo']['name']
#         print(name_user)  

#         # Reading the training samples
#         train_samples = user['trainingSamples']
        
#         # Preprocessing
#         train_segment_X = [get_x_train(user,sample) for sample in train_samples]  
        
#         # Finding the EMG that is the center of each class
#         centers = bestCenter_Class(train_segment_X)
        
#         # Feature extraction by computing the DTW distance between each training
#         # example and the center of each cluster           
#         # Preprocessing the feature vectors
#         X_train = getFeatureExtraction(train_segment_X, centers)
        
#         # Training the feed-forward NN
#         y_train = decode_targets(get_y_train(train_samples))
#         X_val, y_val = get_xy_val(X_train, get_y_train(train_samples)) 
        
#         estimator = trainFeedForwardNetwork(X_train, y_train, X_val, y_val)

#         # Reading the testing samples    
#         test_samples = user['testingSamples']  
        
#         # Concatenating the predictions of all the users for computing the
#         # errors
#         results = ([testing_prediction(user, sample, centers, estimator) for sample in test_samples])         
        
#     responses[name_user]['testing'] = recognition_results(results)

           
# with open('responses5.json', 'w') as json_file:
#   json.dump(responses, json_file)             



             





                


            

        

           




            



