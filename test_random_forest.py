# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 13:30:23 2016

@author: Pranav
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 18:27:04 2016

@author: Pranav
"""

import statistics as stats
from Kappa import get_average_kappa
import util
from RandomForest import RandomForest

training_data_file = 'dataset/small_training_set.tsv'
test_data_file = 'dataset/small_test_set.tsv'

classifier_dump = 'test/dumps/classifier_dump'
training_data_dump = 'test/dumps/training_data_dump'
test_data_dump = 'test/dumps/test_data_dump'

#training_data = util.get_training_data(training_data_file)
#util.dump_object(training_data, training_data_dump)
training_data = util.load_object(training_data_dump)
print(training_data.shape)


forest = RandomForest(training_data, 50, stats.median, 10)
util.dump_object(forest, classifier_dump)

#test_data = util.get_test_data(test_data_file)
#util.dump_object(test_data, test_data_dump)
test_data = util.load_object(test_data_dump)
predictions = []
targets = []

print(test_data.shape)
for sample in test_data:
    targets.append(sample[-1])
    predictions.append(forest.predict(sample[:-1]))
    
print(get_average_kappa(targets, predictions))