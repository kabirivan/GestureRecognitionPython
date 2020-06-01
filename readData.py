import json
import pandas as pd



df = pd.read_json('user1.json')
a = df['synchronizationGesture']['sample1']['accelerometer']['x']
b = df['synchronizationGesture']['sample1']['accelerometer']['x']
c = df['synchronizationGesture']['sample1']['accelerometer']['x']



e =[]
    
for i in range(1,6):       
    a = df['synchronizationGesture']['sample%s' %i]['accelerometer']['x']
    
    e.append(a)
    
    
    
df["new_column"] = df.apply(lambda x: fxy(x['A'], 100), axis=1)