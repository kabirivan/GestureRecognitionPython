#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 14:11:12 2020

@author: aguasharo
"""



# detectar indices en un array
# x = [3,1,7,0,3,1,5,3,2,6,7,3,0]
# y = [3, 5, 6, 7, 8, 9, 10, 22, 1, 2, 3, 4,6]
# get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
# print(get_indexes(1,x))
# a = get_indexes(1,x)         
# idx_Start = y[a[0]]
# print(idx_Start)


# Comprobar uso del espectrograma (Not Working)
# import numpy as np
# from matplotlib import mlab

# data = range(1,1000) #Dummy data. Just for testing

# Fs = 8000
# tWindow = 64e-3
# NWindow = Fs*tWindow
# window = np.hamming(NWindow)

# NFFT = 512
# NOverlap = NWindow/2

# [s, f, t] = mlab.specgram(data, NFFT = NFFT, Fs = Fs, window = window, noverlap = NOverlap, mode='complex')


# Spectrograma con pyfftw



import numpy as np
import matplotlib.pyplot as plt
import simplespectral as sp
from scipy import signal
import pandas as pd




# data = range(1,1000) #Dummy data. Just for testing
# data1 = pd.Series(range(1,1000))

# Fs = 8000
# tWindow = 64e-3
# NWindow = Fs*tWindow
# window = np.hamming(NWindow)
# window1 = signal.hamming(int(NWindow))

# hola = 512
# NOverlap = NWindow/2
# NOverlap1 = int(NWindow/2)

# [s, f, t, im] = plt.specgram(data, NFFT = hola, Fs = Fs, window = window, noverlap = NOverlap)

# ss = abs(s) 



# Add elements into dictionary

# entries_list = {
#  'entry1': 
#  {'key1': 1, 
#   'key2': 2, 
#   'key3': 3,
#   }, 
#  'entry2': 
#   {'key1': 1, 
#    'key2': 2, 
#    'key3': 3,
#    }, 
#  'entry3': 
#    {'key1': 1, 
#     'key2': 2, 
#     'key3': 3,
#     },
#    }

# case_list = []
# for entry in entries_list:
#    case = {'key1': 1, 'key2': 2, 'key3':3 }
#    case_list.append(case)
#    print(case_list)


import numpy as np

## A noisy sine wave as query
idx = np.linspace(0,6.28,num=100)
query = np.sin(idx) + np.random.uniform(size=100)/10.0

## A cosine is for template; sin and cos are offset by 25 samples
template = np.cos(idx)

## Find the best match with the canonical recursion formula
from dtw import *
alignment = dtw(query, template, keep_internals=True)

## Display the warping curve, i.e. the alignment curve
alignment.plot(type="threeway")


