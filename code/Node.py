# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 22:07:18 2016

@author: Pranav
"""

class Node:
    def __init__(self, column = None, split_value = None, left_child = None, right_child = None, result = None):
        self.column = column
        self.split_value = split_value
        self.left_child = left_child
        self.right_child = right_child
        self.result = result
    
    def display(self, level = 0):
        indent = level*5
        if self.result is None:
            print("-"*indent + str((self.column, self.split_value)) + '?')
        else:
            print("-"*indent + 'Predict : ' + str(self.result))
        if self.left_child is not None:
            self.left_child.display(level+1)
        if self.right_child is not None:
            self.right_child.display(level+1)
            
    def predict(self, sample):
        prediction = -1
        if self.result is not None:
            prediction = self.result
        else:
            if sample[self.column] < self.split_value:
                prediction = (self.left_child).predict(sample)
            else:
                prediction = (self.right_child).predict(sample)
        return prediction