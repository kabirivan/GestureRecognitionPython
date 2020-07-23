#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 23:54:28 2020

@author: aguasharo
"""


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