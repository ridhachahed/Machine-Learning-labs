# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    # ***************************************************
    indices=np.arange(len(y))
    np.random.shuffle(indices)
    
    splits=[int(len(y)*ratio)]      #.cum() if severals
    
    x_train,x_test=np.split(x[indices],splits)
    y_train,y_test=np.split(y[indices],splits)
    
    return x_train,y_train,x_test,y_test
    # ***************************************************