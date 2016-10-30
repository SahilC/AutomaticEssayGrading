# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 20:49:22 2016

@author: Pranav
"""
from Features import Features

class Essay:
    score = -1
    essay_set = 0
    essay_id = 0
    features = 0

    @staticmethod
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
    
    def __init__(self, row, store_score=False):
        self.essay_id = row[0]
        self.essay_set = int(row[1])
        text = row[2]
        if store_score:
            self.score =  self.get_score(self.essay_set, row)
        self.features = Features(text)
    