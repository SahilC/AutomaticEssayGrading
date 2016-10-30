# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 23:06:36 2016

@author: Pranav
"""

import statistics as stats
from Kappa import get_average_kappa
import util
from RandomForest import RandomForest

def train_random_forest(training_data_dump):
    training_data = util.load_object(training_data_dump)
    model = RandomForest(training_data, 10, stats.median, 30)
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
    glove_training_data_dump = '../dumps/glove_training_data_dump'
    glove_test_data_dump = '../dumps/glove_test_data_dump'
    
    print('--------------------random forest--------------------')
    model = train_random_forest(glove_training_data_dump)
    avg_kappa = predict_random_forest(model, glove_test_data_dump)
    print('Average quadratic kappa : ' + str(avg_kappa))