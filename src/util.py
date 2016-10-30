# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 23:48:08 2016

@author: Pranav
"""

import codecs
import os
import pickle
import enchant
from Essay import Essay
import numpy as np
from pymongo import MongoClient
from progressbar import ProgressBar

def dump_object(obj, dump_file):
    if not os.path.exists(os.path.dirname(dump_file)):
        os.makedirs(os.path.dirname(dump_file))       
    f = open(dump_file, mode='wb');    
    pickle.dump(obj, f)
    f.close();
    
def load_object(load_file):
    f = open(load_file, mode='rb');    
    obj = pickle.load(f)
    f.close();
    return obj

def get_data(data_file):
    feature_vector = []
    fo = codecs.open(data_file, encoding='utf-8')
    lines = fo.readlines()
    fo.close()
    total = 1.0*len(lines)
    bar = ProgressBar().start()
    line = 0
    for each_line in lines:
        row = each_line.split('\n')[0].split('\t')
        vector = []
        # Ignore the heading line
        if line < 1:
            line += 1
            continue
#        if line % 50 == 0:
#            print('Sample: '+str(line))
            
        e = Essay(row, store_score = True)
        f = e.features
        for i in sorted(f.__dict__.keys()):
            vector.append(f.__dict__[i])
        vector.append(e.score)
        feature_vector.append(np.array(vector))
        line += 1
        bar.update(100*line/total)
    bar.finish()
    return np.array(feature_vector)

def get_score(essay_set, row):
    score = 0
    score = float(row[6])
    if essay_set == 1:
        div = 12
    elif essay_set == 2:
        div = 5
    elif essay_set == 3:
        div = 3
    elif essay_set == 4:
        div = 3
    elif essay_set == 5:
        div = 4
    elif essay_set == 6:
        div = 4
    elif essay_set == 7:
        div = 25
    elif essay_set == 8:
        div = 50
    return score/div

def build_essay_model(collection,essay):
    words = [i for i in essay.lower().split()]
    essay_vector = np.array([0.0 for i in xrange(300)])
    word_list = []
    word_list = words
    for i in collection.find({"gram":{"$in":word_list}}):
        word_vector = [float(n) for n in i['glove_vector']]

        if(len(word_vector) == 300):
            essay_vector += word_vector
    return (essay_vector/len(words))

def get_glove_data(data_file):
    client = MongoClient('localhost', 27017)
    db = client['nlprokz']
    glove = db.glove
    feature_vector = []
    scores = []
    fo = codecs.open(data_file, encoding='utf-8')
    lines = fo.readlines()
    fo.close()
    total = len(lines)*1.0
    line = 0
    bar = ProgressBar().start()
    for each_line in lines:
        line += 1
        if line == 1:
            continue
#        if line % 50 == 0:
#            print "Line - "+str(line)
        row = each_line.split('\n')[0].split('\t')
        essay_set = int(row[1])
        scores.append(get_score(essay_set,row))
        vector = build_essay_model(glove,each_line)
        vector = np.append(vector, get_score(essay_set,row))        
        feature_vector.append(vector)
        bar.update(100*line/total)
    bar.finish()
    return np.array(feature_vector)



