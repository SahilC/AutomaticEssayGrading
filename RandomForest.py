# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 22:47:23 2016

@author: Pranav
"""

from Node import Node
from random import shuffle
from build_tree import *

class RandomForest:
    @staticmethod
    def create_trees(in_data, num_trees, splitting_function, min_data):
        data = np.array(list(in_data))
        shuffle(data)
        length = len(data)
        section_length = length/num_trees
        trees = []
        for i in range(0, num_trees):
            start = i * section_length
            end = start + section_length
            if i == (num_trees - 1):
                section = data[start:, :]
            else:
                section = data[start:end, :]
            trees.append(build_tree(section, splitting_function, min_data))
        return trees
            
    def __init__(self, data, num_trees = 10, splitting_function = stats.median, min_data = 5):
        self.num_trees = num_trees
        self.data = data
        self.splitting_function = splitting_function
        self.min_data = min_data
        self.trees = RandomForest.create_trees(data, num_trees, splitting_function, min_data)
        
    def predict(self, sample):
        prediction = 0
        for t in self.trees:
            prediction += t.predict(sample)
        prediction /= (1.0 * self.num_trees)
        return prediction