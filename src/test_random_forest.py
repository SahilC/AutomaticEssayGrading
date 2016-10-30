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

def train_random_forest(training_data_dump):
    training_data = util.load_object(training_data_dump)
    model = RandomForest(training_data, 50, stats.median, 50)
    return model

def predict_random_forest(model, test_data_dump):
    test_data = util.load_object(test_data_dump)
    predictions = []
    targets = []
    for sample in test_data:
        targets.append(sample[-1])
        predictions.append(model.predict(sample[:-1]))
    return get_average_kappa(targets, predictions)

if __name__ == '__main__':
    training_data_dump = '../dumps/training_data_dump'
    test_data_dump = '../dumps/test_data_dump'
    
    print('--------------------random forest--------------------')
    model = train_random_forest(training_data_dump)
    avg_kappa = predict_random_forest(model, test_data_dump)
    print('Average quadratic kappa : ' + str(avg_kappa))