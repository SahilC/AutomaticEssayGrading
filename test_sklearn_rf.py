# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 18:27:04 2016

@author: Pranav
"""

import codecs
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from Essay import Essay
from Kappa import get_average_kappa
import util

training_data_file = 'dataset/small_training_set.tsv'
test_data_file = 'dataset/small_test_set.tsv'

classifier_dump = 'test/dumps/classifier_dump'
training_data_dump = 'test/dumps/training_data_dump'
test_data_dump = 'test/dumps/test_data_dump'

training_data = util.load_object(training_data_dump)
model = RandomForestRegressor()
d = training_data[:,:-1]
print(d.shape)
t = training_data[:,-1]
print(t.shape)
model = model.fit(d, t)

test_data = util.load_object(test_data_dump)
predictions = []
targets = []

print(test_data.shape)
targets = test_data[:, -1]
predictions = model.predict(test_data[:,:-1])
    
print(get_average_kappa(targets, predictions))