# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 18:55:38 2016

@author: Pranav
"""

import skll.metrics as metric
import numpy as np
import math
import statistics as stats

# Calculates the average quadratic kappa for the entire essay set
def get_average_kappa(targets, predictions):
	num = len(targets)
	total_kappa = 0

	for i in xrange(0, num):
		total_kappa += metric.kappa([targets[i]], [predictions[i]], 'quadratic')

	avg_kappa  = float(total_kappa) / float(num)
	return avg_kappa
 
def avg_minimum_squared_error(targets, predictions):
    length = len(targets)
    assert(length == len(predictions))
    assert(type(targets)==np.ndarray and type(predictions) == np.ndarray)
    
    diff = 1.0*(targets - predictions)
    mse = sum(diff**2)/2
    return mse/length

def avg_chi_square(targets, predictions):
    length = len(targets)
    assert(length == len(predictions))
    assert(type(targets)==np.ndarray and type(predictions) == np.ndarray)
    
    diff = 1.0*(targets - predictions)
    chi_square = sum(diff**2 / np.sqrt(targets))
    return chi_square/length
    
# Data is a vector (1xn or nx1)
def get_variance(data):
    assert(type(data) == np.ndarray)
    mean = stats.mean(data)
    length = 1.0*len(data)
    diff = data - mean
    return sum(diff**2)/length
        
def get_weighted_variance(variance_1, length_1, variance_2, length_2):
    total_length = 1.0*(length_1 + length_2)
    prob_1 = length_1/total_length
    prob_2 = length_2/total_length
    weighted_var = variance_1*prob_1 + variance_2*prob_2
    return weighted_var
     