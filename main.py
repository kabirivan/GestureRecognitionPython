#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 21:06:52 2020

@author: aguasharo
"""
from __future__ import print_function

import json
import os
import itertools
import pandas as pd
import math
from scipy import signal
import numpy as np
import simplespectral as sp
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.manifold import TSNE
import seaborn as sns

import time


import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.optimizers import Adam, SGD

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


import random
from sklearn.preprocessing import StandardScaler

from collections import Counter



#%% Functions



def get_y_train(train_samples):
    
    y_train = []
    
    for sample in train_samples:
        
        y = train_samples[sample]['gestureName']
        
        if y == 'noGesture':
            
            code = 1
            
        elif y == 'fist':
            
            code = 2
            
        elif y == 'waveIn':
            
            code = 3
            
        elif y == 'waveOut':
            
            code = 4
            
        elif y == 'open':
            
            code = 5
            
        elif y == 'pinch':
            
            code = 6
                
       
        y_train.append(code)
        

    return y_train


def butter_lowpass_filter(data, fs, order):
    # Get the filter coefficients 
    b, a = signal.butter(order, fs, 'low', analog = False)
    y = signal.filtfilt(b, a, data)
    return y


def preProcessEMGSegment(EMGsegment_in):
    
    EMG = max(EMGsegment_in)
    
    if EMG > 1:
        EMGnormalized = EMGsegment_in/128
    else:
        EMGnormalized = EMGsegment_in    
             
    EMGrectified = abs(EMGnormalized)   
    EMGsegment_out = butter_lowpass_filter(EMGrectified, 0.1, 5)
    
    
    return EMGsegment_out


def detectMuscleActivity(emg_sum):

    fs = 200
    minWindowLength_Segmentation =  100
    hammingWdw_Length = np.hamming(25)
    numSamples_lapBetweenWdws = 10
    threshForSum_AlongFreqInSpec = 0.85

    [s, f, t, im] = plt.specgram(emg_sum, NFFT = 25, Fs = fs, window = hammingWdw_Length, noverlap = numSamples_lapBetweenWdws, mode = 'magnitude', pad_to = 50)  
    sumAlongFreq = [sum(x) for x in zip(*s)]

    greaterThanThresh = []
    # Thresholding the sum sumAlongFreq
    for item in sumAlongFreq:
        if item >= threshForSum_AlongFreqInSpec:
            greaterThanThresh.append(1)
        else:
            greaterThanThresh.append(0)
           
    greaterThanThresh.insert(0,0)       
    greaterThanThresh.append(0)    
    diffGreaterThanThresh = abs(np.diff(greaterThanThresh)) 

    if diffGreaterThanThresh[-1] == 1:
        diffGreaterThanThresh[-2] = 1;      
       
    x = diffGreaterThanThresh[0:-1];
    findNumber = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    idxNonZero = findNumber(1,x)
    numIdxNonZero = len(idxNonZero)
    idx_Samples = np.floor(fs*t)

    if numIdxNonZero == 0:
        idx_Start = 1
        idx_End = len(emg_sum)
    elif numIdxNonZero == 1:
        idx_Start = idx_Samples[idxNonZero]
        idx_End = len(emg_sum)
    else:
        idx_Start = idx_Samples[idxNonZero[0]]
        idx_End = idx_Samples[idxNonZero[-1]-1]

    numExtraSamples = 25
    idx_Start = max(1,idx_Start - numExtraSamples)
    idx_End = min(len(emg_sum), idx_End + numExtraSamples)
    
    if (idx_End - idx_Start) < minWindowLength_Segmentation:
        idx_Start = 1
        idx_End = len(emg_sum)


    return int(idx_Start), int(idx_End)


def findCentersClass(emg_filtered,sample):
    distances = []
    column = np.arange(0,sample)
    mtx_distances = pd.DataFrame(columns = column)
    mtx_distances = mtx_distances.fillna(0) # with 0s rather than NaNs
    
    
    for sample_i in emg_filtered:
        for sample_j in emg_filtered:   
            dist, dummy = fastdtw(sample_i, sample_j, dist = euclidean)
            distances.append(dist)
            
        df_length = len(mtx_distances)
        mtx_distances.loc[df_length] = distances 
        distances= []  
    vector_dist = mtx_distances.sum(axis=0)
    idx = vector_dist.idxmin()
    center_idx = emg_filtered[int(idx)]
    
    return center_idx


def featureExtraction(emg_filtered, centers):

    dist_features = []
    
    column = np.arange(0,len(centers))
    dataX = pd.DataFrame(columns = column)
    dataX = dataX.fillna(0)
    
    for rep in emg_filtered:
        for middle in centers:
            dist, dummy = fastdtw(rep, middle, dist = euclidean) 
            dist_features.append(dist)
        
        dataX_length = len(dataX)
        dataX.loc[dataX_length] = dist_features
        dist_features = [] 
    
    return dataX

def preProcessFeatureVector(dataX_in):
    
    dataX_mean = dataX_in.mean(axis = 1)
    dataX_std = dataX_in.std(axis = 1)   
    dataX_mean6 =  pd.concat([dataX_mean]*6, axis = 1)
    dataX_std6 =  pd.concat([dataX_std]*6, axis = 1)   
    dataX6 = (dataX_in - dataX_mean6)/dataX_std6
    
    return dataX6


def trainFeedForwardNetwork(X_train,y_train):
    
    classifier = Sequential()
    
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = None, input_dim = 6))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'tanh'))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    classifier.fit(X_train, y_train, batch_size = 150, epochs = 1500,verbose = 2 )
    
    return classifier


def majorite_vote(data, before, after):
    
    votes =[0,0,0,0,0,0]
    class_maj = []
        
    for j in range(0,len(data)):
        wind_mv = data[max(0,(j-before)):min(len(data),(j+after))]
        
        for k in range(0, len(gestures)):
            a = [1 if i == k+1 else 0 for i in wind_mv]  
            votes[k] = sum(a)
            
        findNumber = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
        idx_label = findNumber(max(votes),votes)
        class_maj.append( idx_label[0] + 1)
        
    
    return class_maj


def classifyEMG_SegmentationNN(dataX_test, centers, model):
    sc = StandardScaler()
    window_length = 500
    stride_length = 10
    emg_length = len(dataX_test)
    predLabel_seq = []
    vecTime = []
    timeSeq = []
    
    
    count = 0
    while True:
        start_point = stride_length*count + 1
        end_point = start_point + window_length - 1
        
        if end_point > emg_length:
            break
        
        tStart = time.time()
        window_emg = dataX_test.iloc[start_point:end_point]   
        filt_window_emg = window_emg.apply(preProcessEMGSegment)
        window_sum  = filt_window_emg.sum(axis=1)
        idx_start, idx_end = detectMuscleActivity(window_sum)
        t_acq = time.time()-tStart
        
        if (idx_start != 1) & (idx_end != len(window_emg)) & ((idx_end - idx_start) > 125):
            
            tStart = time.time()
            
            filt_window_emg1 = window_emg.apply(preProcessEMGSegment)
            window_emg1 = filt_window_emg1.iloc[idx_start:idx_end]
            
            
            t_filt = time.time() - tStart
            
            tStart = time.time()
            featVector = featureExtraction([window_emg1], centers)
            featVectorP = preProcessFeatureVector(featVector)
            t_featExtra =  time.time() - tStart
            
            tStart = time.time()
            x = model.predict_proba(featVectorP).tolist()
            probNN = x[0]
            max_probNN = max(probNN)
            predicted_labelNN = probNN.index(max_probNN) + 1
            t_classiNN = time.time() - tStart
            
            tStart = time.time()
            if max_probNN <= 0.5:
                predicted_labelNN = 1
            t_threshNN = time.time() - tStart 
            #print(predicted_labelNN)
           
        else:
            
            t_filt = 0
            t_featExtra = 0
            t_classiNN = 0
            t_threshNN = 0
            predicted_labelNN = 1
            #print('1')
            
            
        count = count + 1
        predLabel_seq.append(predicted_labelNN)
        vecTime.append(start_point+(window_length/2)+50)
        timeSeq.append(t_acq + t_filt + t_featExtra + t_classiNN + t_threshNN)    
    
    pred_seq = majorite_vote(predLabel_seq, 4, 4)    
        
    return  pred_seq, vecTime, timeSeq



def unique(list1): 
    # insert the list to the set 
    list_set = set(list1) 
    # convert the set to the list 
    unique_list = (list(list_set)) 
    
    return unique_list 


def post_ProcessLabels(predicted_Seq):
    
    time_post = []
    predictions = predicted_Seq.copy()
    predictions[0] = 1
    postProcessed_Labels = predictions.copy()
        
    for i in range(1,len(predictions)):
        
        tStart = time.time()
        
        if predictions[i] == predictions[i-1]:
            cond = 1
        else:    
            cond = 0
            
        postProcessed_Labels[i] =  (1 * cond) + (predictions[i]* (1 - cond))
        t_post = time.time() - tStart
        time_post.append(t_post)
        
    time_post.insert(0,time_post[0])     
    uniqueLabels = unique(postProcessed_Labels)
    
    an_iterator = filter(lambda number: number != 1, uniqueLabels)
    uniqueLabelsWithoutRest = list(an_iterator)
       
    if not uniqueLabelsWithoutRest:
        
        finalLabel = 1
        
    else:
        
        if len(uniqueLabelsWithoutRest) > 1:
            finalLabel = uniqueLabelsWithoutRest[0]
            
        else:
            finalLabel = uniqueLabelsWithoutRest[0]
                   
    
    return finalLabel, time_post



#%% Read user data


folderData = 'trainingJSON'
files = []

for root, dirs, files in os.walk(folderData):
     print('Dataset Ready !')
         
         
file_selected = root + '/' + files[2]  

with open(file_selected) as file:
    user = json.load(file)     

# Training Process
train_samples = user['trainingSamples']
num_samples = 25
num_gestures = 6



train_FilteredX = []
train_aux = []
centers = []
counter = 0


for gestures in train_samples:
    
    x = (train_samples[gestures]['emg'])
    df = pd.DataFrame.from_dict(x) / 128
    df = df.apply(preProcessEMGSegment)
    
    df_sum  = df.sum(axis=1)
    idx_Start, idx_End = detectMuscleActivity(df_sum)
    df_seg = df.iloc[idx_Start:idx_End]
    
    train_aux.append(df_seg)
    
    counter = counter + 1
    
    if counter == num_samples:
        print('Gesturee')
        center_gesture = findCentersClass(train_aux,num_samples)
        centers.append(center_gesture)
        counter = 0
        train_aux = []
    
    train_FilteredX.append(df_seg)    

    
features = featureExtraction(train_FilteredX, centers)
X_train = preProcessFeatureVector(features)











targets = get_y_train(train_samples)


y_train = decode_targets(targets)




def decode_targets(y_train):
    
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    target = np_utils.to_categorical(encoded_Y)
    
    return target    
    



estimator = trainFeedForwardNetwork(X_train, y_train)




# for move in gestures:   
#     for i in range(1,26):     
#         sample = train_samples[move]['sample%s' %i]['emg']
#         df = pd.DataFrame.from_dict(sample)
#         df = df.apply(preProcessEMGSegment)
        
#         if segmentation == True:
#             df_sum  = df.sum(axis=1)
#             idx_Start, idx_End = detectMuscleActivity(df_sum)
#         else:
#             idx_Start = 0;
#             idx_End = len(df)
            
#         df_seg = df.iloc[idx_Start:idx_End]   
#         train_aux.append(df_seg)
#         train_FilteredX.append(df_seg)
        
#     center_gesture = findCentersClass(train_aux,25)
#     centers.append(center_gesture)    
#     train_aux = []


# dataX = featureExtraction(train_FilteredX, centers)
# dataX6 = preProcessFeautureVector(dataX)

# X_train = dataX6




   
# y_train = np.array(dataY)
# encoder = LabelEncoder()
# encoder.fit(y_train)
# encoded_Y = encoder.transform(y_train)
# dummy_y = np_utils.to_categorical(encoded_Y)
# estimator = trainFeedForwardNetwork(X_train,dummy_y)




















