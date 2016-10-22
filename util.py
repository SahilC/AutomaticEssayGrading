# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 23:48:08 2016

@author: Pranav
"""

import codecs
import os
import pickle
from Essay import Essay
import numpy as np

def dump_object(obj, dump_file):
    if not os.path.exists(os.path.dirname(dump_file)):
        os.makedirs(os.path.dirname(dump_file))       
    f = open(dump_file, mode='w');    
    pickle.dump(obj, f)
    f.close();
    
def load_object(load_file):
    f = open(load_file, mode='r');    
    obj = pickle.load(f)
    f.close();
    return obj
    
def get_training_data(training_data_file):
#    training_data = 'dataset/small_training_set.tsv'    
    feature_vector = []
    scores = []
    fo = codecs.open(training_data_file, encoding='utf-8')
    lines = fo.readlines()
    fo.close()
    
    line = 0
    for each_line in lines:
        row = each_line.split('\n')[0].split('\t')
        vector = []
        # Ignore the heading line
        if line < 1:
            line += 1
            continue
        if line % 50 == 0:
            print('Training sample: '+str(line))
            
        e = Essay(row, store_score = True)
        f = e.features
        for i in sorted(f.__dict__.keys()):
            vector.append(f.__dict__[i])
        vector.append(e.score)
        scores.append(e.score)
        feature_vector.append(np.array(vector))
        line += 1
    return np.array(feature_vector)

def get_test_data(test_data_file):
#    test_data = 'dataset/small_test_set.tsv'
    feature_vector = []
    scores = []
    fo = codecs.open(test_data_file, encoding='utf-8')
    lines = fo.readlines()
    fo.close()
    
    line = 0
    for each_line in lines:
        row = each_line.split('\n')[0].split('\t')
        vector = []
        # Ignore the heading line
        if line < 1:
            line += 1
            continue
        if line % 50 == 0:
            print('Test sample: '+str(line))
            
        e = Essay(row, store_score = True)
        f = e.features
        for i in sorted(f.__dict__.keys()):
            vector.append(f.__dict__[i])
        vector.append(e.score)
        scores.append(e.score)
        feature_vector.append(np.array(vector))
        line += 1
    return np.array(feature_vector)    