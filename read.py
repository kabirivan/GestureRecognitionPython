#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 01:16:05 2020

@author: aguasharo
"""


import json

with open('user1test.json') as file:
    data = json.load(file)


emg_sample1 = data['synchronizationGesture']['sample1']['emg']

#emg = emg_sample1.get('emg').get('ch1')

signals = []

for item in emg_sample1.values():

   signals.append(item)
    


