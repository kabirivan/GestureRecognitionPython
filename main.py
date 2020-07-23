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

import psutil
import ray

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


def get_y_train(train_samples):
    # Changes a gesture into a code
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



def butter_lowpass_filter(data, fs, order):
    # Get the filter coefficients 
    b, a = signal.butter(order, fs, 'low', analog = False)
    y = signal.filtfilt(b, a, data)
    return y


def preProcessEMGSegment(EMGsegment_in):
    # This function to apply a filter
    EMG = max(EMGsegment_in)
    
    if EMG > 1:
        EMGnormalized = EMGsegment_in/128
    else:
        EMGnormalized = EMGsegment_in    
             
    EMGrectified = abs(EMGnormalized)   
    EMGsegment_out = butter_lowpass_filter(EMGrectified, 0.1, 5)
    
    
    return EMGsegment_out


def detectMuscleActivity(emg_sum):
    
    # This function segments in a EMG the region corresponding to a muscle
    # contraction. The indices idxStart and idxEnd correspond to the begining
    # and the end of such a region

    # Sampling frequency of the EMG
    fs = 200
    minWindowLength_Segmentation =  100 # Minimum length of the segmented region
    hammingWdw_Length = np.hamming(25) # Window length
    numSamples_lapBetweenWdws = 10 # Overlap between 2 consecutive windows
    threshForSum_AlongFreqInSpec = 0.86

    [s, f, t, im] = plt.specgram(emg_sum, NFFT = 25, Fs = fs, window = hammingWdw_Length, noverlap = numSamples_lapBetweenWdws, mode = 'magnitude', pad_to = 50)  
    
    # Summing the spectrogram along the frequencies
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
    # Finding the indices of the start and the end of a muscle contraction
    if numIdxNonZero == 0:
        idx_Start = 1
        idx_End = len(emg_sum)
    elif numIdxNonZero == 1:
        idx_Start = idx_Samples[idxNonZero]
        idx_End = len(emg_sum)
    else:
        idx_Start = idx_Samples[idxNonZero[0]]
        idx_End = idx_Samples[idxNonZero[-1]-1]
    # Adding a head and a tail to the segmentation
    numExtraSamples = 25
    idx_Start = max(1,idx_Start - numExtraSamples)
    idx_End = min(len(emg_sum), idx_End + numExtraSamples)
    
    if (idx_End - idx_Start) < minWindowLength_Segmentation:
        idx_Start = 1
        idx_End = len(emg_sum)


    return int(idx_Start), int(idx_End)


def EMG_segment(train_filtered_X):
    # This function return a segment with corresponding to a muscle
    # contraction
    
    df_sum  = train_filtered_X.sum(axis=1)
    idx_Start, idx_End = detectMuscleActivity(df_sum)
    df_seg = train_filtered_X.iloc[idx_Start:idx_End]
    
    return df_seg 



def findCentersClass(emg_filtered):
    # This function returns a set of time series called centers. The ith
    # time series of centers, centers{i}, is the center of the cluster of time 
    # series from the set timeSeries that belong to the ith class. For finding
    # the center of each class, the DTW distance is used.  

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


def featureExtraction(emg_filtered, centers):
    # This function computes a feature vector for each element from the set
    # timeSeries. The dimension of this feature vector depends on the number of 
    # time series of the set centers. The value of the jth feature of the ith
    # vector in dataX corresponds to the DTW distance between the signals 
    # timeSeries{i} and centers{j}.  

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
    # This function preprocess each feature vector of the set dataX_in. Each
    # row of dataX_in is a fetaure vector and each column is a featur
    
    dataX_mean = dataX_in.mean(axis = 1)
    dataX_std = dataX_in.std(axis = 1)   
    dataX_mean6 =  pd.concat([dataX_mean]*num_gestures, axis = 1)
    dataX_std6 =  pd.concat([dataX_std]*num_gestures, axis = 1)   
    dataX6 = (dataX_in - dataX_mean6)/dataX_std6
    
    return dataX6


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


def majorite_vote(data, before, after):
    # This function is used to apply pos-processing based on majority vote
    
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
    # This function applies a hand gesture recognition model based on artificial
    # feed-forward neural networks and automatic feature extraction to a set of
    # EMGs conatined in the set test_X. The actual label of each EMG in test_X
    # is in the set test_Y. The structure nnModel contains the trained neural
    # network     
    
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
    # This function post-processes the sequence of labels returned by a
    # classifier. Each row of predictedSeq is a sequence of 
    # labels predicted by a different classifier for the jth example belonging
    # to the ith actual class.
    
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



def code2gesture(code):
    # This function returns the gesture name from code
    
        
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
                       
        
    return label


def code2gesture_labels(vector_labels_prev):
    # This function returns a prediction vector with gesture names
    
    v2 = []
    
    for window in vector_labels_prev:
        
        vec_prev = code2gesture(window)        
        v2.append(vec_prev)
    
    return v2    


def classify_gesture(test_RawX, centers, estimator):
    
    # This function applies a hand gesture recognition model based on artificial
    # feed-forward neural networks and automatic feature extraction to a set of
    # EMGs conatined in the set test_RawX
    
    [predictedSeq, vec_time, time_seq]= classifyEMG_SegmentationNN(test_RawX, centers, estimator)
    predicted_label, t_post = post_ProcessLabels(predictedSeq)
    
    # Computing the time of processing
    estimatedTime =  [sum(x) for x in zip(time_seq, t_post)]
       
    return predicted_label, predictedSeq, vec_time, estimatedTime


def testing_prediction(user,sample):
    
    
    test_RawX = get_x_test(user,sample) 
    predicted_label, predictedSeq, vec_time, estimatedTime = classify_gesture(test_RawX, centers, estimator)
    
    return predicted_label, predictedSeq, vec_time, estimatedTime



def recognition_results(results):
    
     # This function save the responses of each user into a dictionary

    d = collections.defaultdict(dict)
    
    for i in range(0,150):
                
        d['idx_%s' %i]['class'] = code2gesture(results[i][0])
        d['idx_%s' %i]['vectorOfLabels'] = code2gesture_labels(results[i][1])
        d['idx_%s' %i]['vectorOfTimePoints'] = results[i][2]
        d['idx_%s' %i]['vectorOfProcessingTimes']= results[i][3]    
       
    return d



#%% Read user data
responses = collections.defaultdict(dict)
num_gestures = 6
folderData = 'trainingJSON'
files = []



for root, dirs, files in os.walk(folderData):
     print('Dataset Ready !')
         
     
for user_data in files:
    file_selected = root + '/' + user_data 
    with open(file_selected) as file:
        user = json.load(file)      
        name_user = user['userInfo']['name']
        print(name_user)  

        # Reading the training samples
        train_samples = user['trainingSamples']
        
        # Preprocessing
        train_segment_X = [get_x_train(user,sample) for sample in train_samples]  
        
        # Finding the EMG that is the center of each class
        centers = bestCenter_Class(train_segment_X)
        
        # Feature extraction by computing the DTW distance between each training
        # example and the center of each cluster     
        features = featureExtraction(train_segment_X, centers)
        
        # Preprocessing the feature vectors
        X_train = preProcessFeatureVector(features)
        
        # Training the feed-forward NN
        y_train = decode_targets(get_y_train(train_samples))
        X_val, y_val = get_xy_val(X_train, get_y_train(train_samples)) 
        
        estimator = trainFeedForwardNetwork(X_train, y_train, X_val, y_val)

        # Reading the testing samples    
        test_samples = user['testingSamples']  
        
        # Concatenating the predictions of all the users for computing the
        # errors
        results = ([testing_prediction(user,sample) for sample in test_samples])         
        
    responses[name_user]['testing'] = recognition_results(results)


           
with open('responses5.json', 'w') as json_file:
  json.dump(responses, json_file)             




             





                


            

        

           




            



