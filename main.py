#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 21:06:52 2020

@author: aguasharo
"""


import json
import os

folderData = 'trainingJSON'
gestures = ['noGesture', 'open', 'fist', 'waveIn', 'waveOut', 'pinch']


files = []

for root, dirs, files in os.walk(folderData):
     print('Dataset Ready !')
         
         
file_selected = root + '/' + files[0]  

with open(file_selected) as file:
    user = json.load(file)     

# Training Process
train_samples = user['trainingSamples']
train_noGesture = train_samples.get('noGesture')
train_fist = train_samples.get('fist')
train_open= train_samples.get('open')
train_pinch = train_samples.get('pinch')
train_waveIn = train_samples.get('waveIn')
train_waveOut = train_samples.get('waveOut')


for item in train_samples.values():
   train_gesture.append(item)









# for i in 

# 'P{} Pressure [mmHg]'.format(j)










