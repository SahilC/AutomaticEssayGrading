# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 18:27:04 2016

@author: Pranav
"""

from sklearn.ensemble import RandomForestRegressor
from Kappa import get_average_kappa
import util

def train_sklearn_random_forest(training_data_dump):
    training_data = util.load_object(training_data_dump)
    model = RandomForestRegressor(n_estimators = 10)
    model = model.fit(training_data[:,:-1], training_data[:,-1])
    return model

def predict_sklearn_random_forest(model, test_data_dump):
    test_data = util.load_object(test_data_dump)
    predictions = []
    targets = []
    
    targets = test_data[:, -1]
    predictions = model.predict(test_data[:,:-1])    
    return get_average_kappa(targets, predictions)

if __name__ == '__main__':
    glove_training_data_dump = 'test/dumps/glove_training_data_dump'
    glove_test_data_dump = 'test/dumps/glove_test_data_dump'
    
    print('--------------------sklearn random forest regressor--------------------')
    model = train_sklearn_random_forest(training_data_dump)
    avg_kappa = predict_sklearn_random_forest(model, test_data_dump)
    print('Average quadratic kappa : ' + str(avg_kappa))