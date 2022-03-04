#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import networkx as nx

def threshold_output(W, desired_edges=None, verbose=False):
    if desired_edges != None:
        # Implements binary search for acyclic graph with the desired number of edges
        ws = sorted([abs(i) for i in np.unique(W)])
        best = W
        done = False
        
        mid = int(len(ws)/2.0)
        floor = 0
        ceil = len(ws)-1
        
        while not done:
            cut = np.array([[0.0 if abs(i) <= ws[mid] else 1.0 for i in j] for j in W])
            g = nx.from_numpy_array(cut, create_using=nx.DiGraph())
            try:
                nx.find_cycle(g)
                floor = mid
                mid = int((ceil-mid)/2.0) + mid
                if mid == ceil or mid == floor:
                    done = True
            except:
                if nx.number_of_edges(g) == desired_edges:
                    best = cut
                    done = True
                elif nx.number_of_edges(g) >= desired_edges:
                    best = cut
                    floor = mid
                    mid = int((ceil-mid)/2.0) + mid
                    if mid == ceil or mid == floor:
                        done = True
                else:
                    best = cut
                    ceil = mid
                    mid = int((floor-mid)/2.0) + floor
                    if mid == ceil or mid == floor:
                        done = True
    else:
        ws = sorted([abs(i) for i in np.unique(W)])
        best = None

        for w in ws:
            cut = np.array([[0.0 if abs(i) <= w else 1.0 for i in j] for j in W])
            g = nx.from_numpy_array(cut, create_using=nx.DiGraph())
            try:
                nx.find_cycle(g)
            except:
                return cut
    return best

