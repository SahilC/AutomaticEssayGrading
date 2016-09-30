# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 22:07:18 2016

@author: Pranav
"""

class Node:
    def __init__(self, column, value, left_child, right_child):
        self.column = column
        self.value = value
        self. left_child = left_child
        self.right_child = right_child
        self.result = -1