# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 23:48:08 2016

@author: Pranav
"""

import os
import pickle

def dump_classifier(classifier, dump_file):
    if not os.path.exists(dump_file):
        os.makedirs(os.path.dirname(dump_file))       
    f = open(dump_file, mode='a');    
    pickle.dump(classifier, f)
    f.close();
    
def load_classifier(load_file):
    f = open(load_file, mode='a');    
    classifier = pickle.load(f)
    f.close();
    return classifier