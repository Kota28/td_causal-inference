#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
def SHD(target, pred, double_for_anticausal=True):
    import numpy as np
    diff = np.abs(target - pred)
    if double_for_anticausal:
        return np.sum(diff)
    else:
        diff = diff + diff.transpose()
        diff[diff > 1] = 1  # Ignoring the double edges.
        return np.sum(diff)/2

