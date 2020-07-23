#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 23:22:24 2020

@author: aguasharo
"""

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


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