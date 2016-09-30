# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 22:39:23 2016

@author: Pranav
"""

import statistics as stats
import numpy as np
from Node import Node

# Takes the data matrix and a column number to split on.
# Splitting function is used to calculate the value to split on, e.g. mean/median
# Returns the data split into two parts
def split_data(data, column, splitting_function):
    value = splitting_function(data[:, column])
    print(value)
    part_one = []
    part_two = []
    
    for row in data:
        if value < row[column]:
            part_one.append(row)
        else:
            part_two.append(row)
    
    return [part_one, part_two]

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

def get_total_variance(part_one, part_two):
    var_1 = get_variance(part_one[:, -1])
    length_1 = part_one.shape[0]
    var_2 = get_variance(part_two[:, -1])
    length_2 = part_two.shape[0]
    total_var = get_weighted_variance(var_1, length_1, var_2, length_2)
    return total_var

# Returns the column to split which produces the least total variance          
def find_best_split(data, splitting_function):
    if len(data) == 0:
        return -1
    cols = data.shape[1]
    best_col = 0
    best_var = 9999999
    for col in range(0,cols-1): # Last column has the score, so leaving it out
        [part_one, part_two] = split_data(data, col, splitting_function)
        cur_var = get_total_variance(part_one, part_two)
        if best_var > cur_var:
            best_var = cur_var
            best_col = col
    return (best_col, best_var)
    
def build_tree(data, splitting_function = stats.mean):
    if len(data) == 0:
        return
    (best_col, best_var) = find_best_split(data)
    [part_one, part_two] = split_data(data, best_col , splitting_function)
    left_branch = build_tree(part_one, splitting_function)
    right_branch = build_tree(part_two, splitting_function)
    return Node(best_col, best_var, left_branch, right_branch)