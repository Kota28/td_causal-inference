#!/usr/bin/env python
# coding: utf-8

# In[93]:


import numpy as np
import pandas as pd
# np.random.seed(2022)

def generate(size, dimension, seed):
    n = size
    d = dimension
    
    np.random.seed(seed)
    B = np.random.randint(-3, 3, (d, d))
    
    # Designate causal order. In this case, 0, 1, 2, ..., d to make it simpler
    B[0,:] = 0 # x_0 is a parant of all variables including indirect parent.
    for i in range(1, d):
        for j in range(i, d):
            B[i, j] = 0
    
    B = B.T
    print("Ground Truth:")
    print(B)
    
    cols = []
    for i in range(d):
        cols.append("x_" + str(i))
    
    df = pd.DataFrame(columns=cols)
    
    for _ in range(n):
        dataset = [0 for _ in range(d)]
        dataset[0] = float(np.random.uniform(size=1))
        for i in range(1, d):
            for j in range(i):
                dataset[i] += B[j, i] * dataset[j] + float(np.random.uniform(size=1))
        df = df.append(pd.Series(dataset, index=df.columns), ignore_index = True)
        
    directlingam_result = []
    notears_result = []
    num_nodes = len(B)
    num_edges = num_nodes
    edge_coefficient_range = [0.5, 2.0]
    return df, B

