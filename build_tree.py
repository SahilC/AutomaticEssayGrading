# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 22:39:23 2016

@author: Pranav
"""

import statistics as stats
import numpy as np

from Node import Node
from Kappa import *

# Takes the data matrix and a column number to split on.
# Splitting function is used to calculate the value to split on, e.g. mean/median
# Returns the data split into two parts
def split_data(data, column, splitting_function):
    split_value = splitting_function(data[:, column])
    part_one = []
    part_two = []
    for row in data:
        if split_value > row[column]:
            part_one.append(row)
        else:
            part_two.append(row)
    
    return [np.array(part_one), np.array(part_two)]

def get_total_variance(part_one, part_two):
    length_1 = part_one.shape[0]
    length_2 = part_two.shape[0]
    var_1 = 0
    var_2 = 0
    if length_1 > 0:
        var_1 = get_variance(part_one[:, -1])
    if length_2 > 0:
        var_2 = get_variance(part_two[:, -1])
    
    total_var = get_weighted_variance(var_1, length_1, var_2, length_2)
    return total_var

# Returns the column to split which produces the least total variance          
def find_best_split(data, splitting_function = stats.median):
    if len(data) == 0:
        return -1
    cols = data.shape[1]
    best_col = 0
    best_var = 9999999
    for col in range(0,cols-1): # Last column has the score, so leaving it out
        [part_one, part_two] = split_data(data, col, splitting_function)
        cur_var = get_total_variance(np.array(part_one), np.array(part_two))
        if best_var > cur_var:
            best_var = cur_var
            best_col = col
    return (best_col, best_var)
    
def build_tree(data, splitting_function = stats.median, min_data = 5):
    assert(min_data >= 1)    
    
    length = 1.0*len(data)
    left_branch = None
    right_branch = None
    
    if len(data) <= min_data:
        result = sum(data[:, -1])/length
        return Node(result = result)
        
    (best_col, best_var) = find_best_split(data, splitting_function)
    split_val = splitting_function(data[:,best_col])
    [part_one, part_two] = split_data(data, best_col , splitting_function)
    
    if len(part_one) == 0 or len(part_two) == 0:
        result = sum(data[:, -1])/length
        return Node(result = result)
    else:    
        left_branch = build_tree(part_one, splitting_function, min_data)
        right_branch = build_tree(part_two, splitting_function, min_data)
        
    return Node(best_col, split_val, left_branch, right_branch)
