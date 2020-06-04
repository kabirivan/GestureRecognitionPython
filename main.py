#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 21:06:52 2020

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
gestures = ['noGesture', 'fist', 'waveIn', 'waveOut', 'open', 'pinch']


files = []

for root, dirs, files in os.walk(folderData):
     print('Dataset Ready !')
         
         
file_selected = root + '/' + files[2]  

with open(file_selected) as file:
    user = json.load(file)     

# Training Process
train_samples = user['trainingSamples']

train_noGesture = []
train_open = []
train_fist = []
train_waveIn = []
train_waveOut = []
train_pinch =[]




def butter_lowpass_filter(data, fs, order):
    # Get the filter coefficients 
    b, a = signal.butter(order, fs, 'low', analog=False)
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



def detectMuscleActivity(emgSignal):    
    fs = 200
    minWindowLength_Segmentation =  100
    hammingWdw_Length = np.hamming(25)
    numSamples_lapBetweenWdws = 10
    threshForSum_AlongFreqInSpec = 10
    # Computing the spectrogram of the EMG
    Sxx, freqs, time, im = plt.specgram(emgSignal, NFFT = 25, Fs = fs, window = hammingWdw_Length, noverlap = numSamples_lapBetweenWdws)
    #Sxx, freqs, time, im = emgSignal.specgram(x = emgSignal, NFFT=FFT, window = numpy.hamming(25), Fs = 200, noverlap = 10)       
    # Summing the spectrogram along the frequencies
    sumAlongFreq = [sum(x) for x in zip(*Sxx)]
    

    
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

    x = diffGreaterThanThresh[1:-2];
    
    findNumber = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    idxNonZero = findNumber(1,x)
    numIdxNonZero = len(idxNonZero)
    idx_Samples = fs*time
    idx_Samples = np.floor(idx_Samples) 
    
    if numIdxNonZero == 0:
        idx_Start = 1
        idx_End = len(emgSignal)
    elif numIdxNonZero == 1:
        idx_Start = idx_Samples[idxNonZero[0]]
        idx_End = len(emgSignal)
    else:
        idx_Start = idx_Samples[idxNonZero[0]]
        idx_End = idx_Samples[idxNonZero[-2]]
    
    numExtraSamples = 25
    idx_Start = max(1,idx_Start - numExtraSamples)
    idx_End = min(len(emgSignal), idx_End + numExtraSamples)
    
    if (idx_End - idx_Start) < minWindowLength_Segmentation:
        idx_Start = 1
        idx_End = len(emgSignal)


    
    return idx_Start, idx_End

 
       

dataY = list(itertools.chain.from_iterable(itertools.repeat(x, 25) for x in range(1,len(gestures)+1)))
segmentation = False
    
for i in range(1,26):       
    train_noGesture.append(train_samples['noGesture']['sample%s' %i]['emg'])
    train_fist.append(train_samples['fist']['sample%s' %i]['emg'])
    train_waveIn.append(train_samples['waveIn']['sample%s' %i]['emg'])
    train_waveOut.append(train_samples['waveOut']['sample%s' %i]['emg'])
    train_open.append(train_samples['open']['sample%s' %i]['emg'])
    train_pinch .append(train_samples['pinch']['sample%s' %i]['emg'])
    
    

 

train_FilteredX = []


for i in range(0,25):
    sample = train_noGesture[i]
    df = pd.DataFrame.from_dict(sample)
    df = df.apply(preProcessEMGSegment)
    
    if segmentation == True:
        df_sum  = df.sum(axis=1)
        idx_Start, idx_End = detectMuscleActivity(df_sum)
    else:
        idx_Start = 0;
        idx_End = len(df)
        
    df.iloc[idx_Start:idx_End]
    
    train_FilteredX.append(df)
    
a = train_FilteredX[0]
b = train_FilteredX[4]

distances = []
c = 0



column = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25']
mtx_distances = pd.DataFrame(columns = column)
mtx_distances = mtx_distances.fillna(0) # with 0s rather than NaNs


for sample_i in train_FilteredX:
    for sample_j in train_FilteredX:   
        dist, dummy = fastdtw(sample_i, sample_j, dist = euclidean)
        distances.append(dist)
        
    df_length = len(mtx_distances)
    mtx_distances.loc[df_length] = distances 
    distances= []  
vector_dist = mtx_distances.sum()
    
        
        
        
    #     dist, dummy = fastdtw(sample_i, sample_j, dist = euclidean)
    #     # print(dist)
    #     mtxDistances_class_i.append([dist])
    # df_ = df_.append(mtxDistances_class_i, ignore_index=True)
    # mtxDistances_class_i = []
    
    
    


















# for i in 

# 'P{} Pressure [mmHg]'.format(j)










