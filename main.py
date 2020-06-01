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


dataY = list(itertools.chain.from_iterable(itertools.repeat(x, 25) for x in range(1,len(gestures)+1)))

    
for i in range(1,26):       
    train_noGesture.append(train_samples['noGesture']['sample%s' %i]['emg'])
    train_fist.append(train_samples['fist']['sample%s' %i]['emg'])
    train_waveIn.append(train_samples['waveIn']['sample%s' %i]['emg'])
    train_waveOut.append(train_samples['waveOut']['sample%s' %i]['emg'])
    train_open.append(train_samples['open']['sample%s' %i]['emg'])
    train_pinch .append(train_samples['pinch']['sample%s' %i]['emg'])
    


sample = train_noGesture[0]
df = pd.DataFrame.from_dict(sample)




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



# def detectMuscleActivity(emgSignal)
#     fs = 200
#     minWindowLength_Segmentation =  100
#     numFreq_Spec = 50;
#     hammingWdw_Length = 25;
#     numSamples_lapBetweenWdws = 10;
#     threshForSum_AlongFreqInSpec = 10
        





df = df.apply(preProcessEMGSegment)
df_sum  = df.sum(axis=1)




        
    





# for i in 

# 'P{} Pressure [mmHg]'.format(j)










