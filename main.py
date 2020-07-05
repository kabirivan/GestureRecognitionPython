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

import collections
from collections import Counter

import multiprocessing as mp
from joblib import Parallel, delayed


import ray

#%% Functions


            
def get_x_train(user,sample):
    train_samples = user['trainingSamples']
    x = (train_samples[sample]['emg'])
    df = pd.DataFrame.from_dict(x) / 128
    
    return df


    




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


def decode_targets(y_train):
    
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    target = np_utils.to_categorical(encoded_Y)
    
    return target    


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
    threshForSum_AlongFreqInSpec = 0.857

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


def EMG_segment(train_filtered_X):
    
    df_sum  = train_filtered_X.sum(axis=1)
    idx_Start, idx_End = detectMuscleActivity(df_sum)
    df_seg = train_filtered_X.iloc[idx_Start:idx_End]
    
    return df_seg 



@ray.remote
def findCentersClass(emg_filtered):
    distances = []
    sample = 25
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
    dataX_mean6 =  pd.concat([dataX_mean]*num_gestures, axis = 1)
    dataX_std6 =  pd.concat([dataX_std]*num_gestures, axis = 1)   
    dataX6 = (dataX_in - dataX_mean6)/dataX_std6
    
    return dataX6


def trainFeedForwardNetwork(X_train,y_train, X_test, y_test):
    
    classifier = Sequential()
    
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = None, input_dim = 6))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'tanh'))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    classifier.fit(X_train, y_train, batch_size = 150, epochs = 1300, validation_data = (X_test, y_test),verbose = 0 )
    
    return classifier


def majorite_vote(data, before, after):
    
    votes =[0,0,0,0,0,0]
    class_maj = []
        
    for j in range(0,len(data)):
        wind_mv = data[max(0,(j-before)):min(len(data),(j+after))]
        
        for k in range(0, 6):
            a = [1 if i == k+1 else 0 for i in wind_mv]  
            votes[k] = sum(a)
            
        findNumber = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
        idx_label = findNumber(max(votes),votes)
        class_maj.append( idx_label[0] + 1)
        
    
    return class_maj


def classifyEMG_SegmentationNN(dataX_test, centers, model):
    sc = StandardScaler()
    window_length = 600
    stride_length = 30
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
        
        if (idx_start != 1) & (idx_end != len(window_emg)) & ((idx_end - idx_start) > 85):
            
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



def code2gesture(vec_class):

    vector = []    
      
    for code in vec_class:
        
        if code == 1:
            
            label = 'noGesture'
            
        elif code == 2:
            
            label = 'fist'
                          
        elif code == 3:
            
            label = 'waveIn'
            
        elif code == 4:
            
            label = 'waveOut'
        
        elif code == 5:
            
            label = 'open'            
    
    
        elif code == 6:
            
            label = 'pinch'
            
            
        vector.append(label)
        
        
    return vector


def code2gesture_labels(vector_class_prev,vector_labels_prev):
    
    
    v1 = code2gesture(vector_class_prev)
    v2 = []
    
    for window in vector_labels_prev:
        
        vec_prev = code2gesture(window)
        
        v2.append(vec_prev)
        
        
    return v1, v2    



def testing(x, centers, estimator):
    
    
    df_test = pd.DataFrame.from_dict(x) / 128
    
    [predictedSeq, vec_time, time_seq]= classifyEMG_SegmentationNN(df_test, centers, estimator)
    predicted_label, t_post = post_ProcessLabels(predictedSeq)
    estimatedTime =  [sum(x) for x in zip(time_seq, t_post)]
    
    vector_ProcessingTimes.append(estimatedTime) 
    
    vector_class, vector_labels = code2gesture_labels(predicted_label, predictedSeq)
    
    return vector_class, vector_labels, estimatedTime, time_seq




#%% Read user data
test = collections.defaultdict(dict)
ray.init()

folderData = 'trainingJSON'
files = []
counter = 0
for root, dirs, files in os.walk(folderData):
     print('Dataset Ready !')
         
         

#%% 


for user_data in files:
    file_selected = root + '/' + user_data 
    with open(file_selected) as file:
        user = json.load(file)   
        
        name_user = user['userInfo']['name']
        print(name_user)  

        train_samples = user['trainingSamples']
        num_samples = 25
        num_gestures = 6
        train_segment_X = []
        train_FilteredX_app = []
        train_aux = []
        centers = []
        counter = 0


        for sample in train_samples:
            
            train_RawX = get_x_train(user,sample)
            train_filtered_X = train_RawX.apply(preProcessEMGSegment)
            train_segment_X.append(EMG_segment(train_filtered_X))
            
            
            
            

    

    
    
        
           
ray.shutdown()           
            
#             df_sum  = df.sum(axis=1)
#             idx_Start, idx_End = detectMuscleActivity(df_sum)
#             df_seg = df.iloc[idx_Start:idx_End]
            
#             train_aux.append(df_seg)
#             train_FilteredX_app.append(df_seg)
            
#             counter = counter + 1
            
#             if counter == num_samples:
#                 print('Gesturee')
#                 center_gesture = findCentersClass.remote(train_aux)
                
                
#                 counter = 0
#                 train_aux = []
            
#         centers = ray.get(center_gesture)
        
        
        
        
#         features = featureExtraction(train_FilteredX_app, centers)     
#         X_train = preProcessFeatureVector(features)
        
#         targets = get_y_train(train_samples)
#         y_train = decode_targets(targets)
        
        
#         data_val = X_train.copy()
#         data_val['6'] = targets
        
#         xy_val = data_val.sample(frac=1).reset_index(drop=True)
        
        
#         X_val = xy_val.iloc[:,0:6]  
#         y_val = decode_targets(xy_val['6'])
        
        
#         estimator = trainFeedForwardNetwork(X_train, y_train, X_val, y_val)


#         vector_class_prev = []
#         vector_TimePoints = []
#         vector_labels_prev = []
#         vector_ProcessingTimes = []
        
#         test_samples = user['testingSamples']
               
            
#         for sample in test_samples:
            
#             x = (test_samples[sample]['emg'])
#             df_test = pd.DataFrame.from_dict(x) / 128
            
#             [predictedSeq, vec_time, time_seq]= classifyEMG_SegmentationNN(df_test, centers, estimator)
#             predicted_label, t_post = post_ProcessLabels(predictedSeq)
#             estimatedTime =  [sum(x) for x in zip(time_seq, t_post)]
            
#             vector_class_prev.append(predicted_label)
#             vector_labels_prev.append(predictedSeq)
#             vector_TimePoints.append(vec_time)  
#             vector_ProcessingTimes.append(estimatedTime) 
            
#             vector_class, vector_labels = code2gesture_labels(vector_class_prev,vector_labels_prev)
            

#         d = collections.defaultdict(dict)
        
        
#         for i in range(0,150):
#             d['idx_%s' %i]['class'] = vector_class[i]
#             d['idx_%s' %i]['vectorOfLabels'] = vector_labels[i]
#             d['idx_%s' %i]['vectorOfTimePoints'] = vector_TimePoints[i]
#             d['idx_%s' %i]['vectorOfProcessingTimes']= vector_ProcessingTimes[i]

        
#     test[name_user]['testing'] = d   


# with open('responses.txt', 'w') as json_file:
#   json.dump(test, json_file)   


#%% Preprocess data

# for user_data in files:
#     file_selected = root + '/' + user_data 
#     with open(file_selected) as file:
#         user = json.load(file)   
        
#         name_user = user['userInfo']['name']
#         print(name_user)  



#         train_samples = user['trainingSamples']
#         num_samples = 25
#         num_gestures = 6
#         train_FilteredX = []
#         train_aux = []
#         centers = []
#         counter = 0


#         for sample in train_samples:
            
#             x = (train_samples[sample]['emg'])
#             df = pd.DataFrame.from_dict(x) / 128
#             df = df.apply(preProcessEMGSegment)
            
#             df_sum  = df.sum(axis=1)
#             idx_Start, idx_End = detectMuscleActivity(df_sum)
#             df_seg = df.iloc[idx_Start:idx_End]
            
#             train_aux.append(df_seg)
            
#             counter = counter + 1
            
#             if counter == num_samples:
#                 print('Gesturee')

#                 center_gesture = findCentersClass(train_aux)
#                 centers.append(center_gesture)
#                 counter = 0
#                 train_aux = []
            
#             train_FilteredX.append(df_seg)
            
            
#         features = featureExtraction(train_FilteredX, centers)
        
#         X_train = preProcessFeatureVector(features)

#         targets = get_y_train(train_samples)
#         y_train = decode_targets(targets)
        
        
#         data_val = X_train.copy()
#         data_val['6'] = targets
        
#         xy_val = data_val.sample(frac=1).reset_index(drop=True)
        
        
#         X_val = xy_val.iloc[:,0:6]  
#         y_val = decode_targets(xy_val['6'])
        
        
#         estimator = trainFeedForwardNetwork(X_train, y_train, X_val, y_val)



#         vector_class_prev = []
#         vector_TimePoints = []
#         vector_labels_prev = []
#         vector_ProcessingTimes = []
        
#         test_samples = user['testingSamples']
        
#         for sample in test_samples:
            
#             x = (test_samples[sample]['emg'])
#             df_test = pd.DataFrame.from_dict(x) / 128
            
#             [predictedSeq, vec_time, time_seq]= classifyEMG_SegmentationNN(df_test, centers, estimator)
#             predicted_label, t_post = post_ProcessLabels(predictedSeq)
#             estimatedTime =  [sum(x) for x in zip(time_seq, t_post)]
            
#             vector_class_prev.append(predicted_label)
#             vector_labels_prev.append(predictedSeq)
#             vector_TimePoints.append(vec_time)  
#             vector_ProcessingTimes.append(estimatedTime) 
            
#             vector_class, vector_labels = code2gesture_labels(vector_class_prev,vector_labels_prev)
            
#             print(sample)
    
                   
# #%%


#         d = collections.defaultdict(dict)
        
        
#         for i in range(0,150):
#             d['idx_%s' %i]['class'] = vector_class[i]
#             d['idx_%s' %i]['vectorOfLabels'] = vector_labels[i]
#             d['idx_%s' %i]['vectorOfTimePoints'] = vector_TimePoints[i]
#             d['idx_%s' %i]['vectorOfProcessingTimes']= vector_ProcessingTimes[i]

        
#     test[name_user]['testing'] = d   


# with open('responses.txt', 'w') as json_file:
#   json.dump(test, json_file)   


# #%%


# print("Number of cpu : ", mp.cpu_count())


