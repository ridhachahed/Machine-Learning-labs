# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    w=np.linalg.solve(tx.T @ tx  + lambda_* 2*len(y) *np.identity(tx.shape[1]) ,tx.T @ y)
    return w
    # ***************************************************
