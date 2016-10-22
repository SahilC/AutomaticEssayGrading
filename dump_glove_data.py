# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 22:42:51 2016

@author: Pranav
"""
import util
test_data_file = 'dataset/small_test_set.tsv'
training_data_file = 'dataset/small_training_set.tsv'

glove_training_data_dump = 'test/dumps/glove_training_data_dump'
glove_test_data_dump = 'test/dumps/glove_test_data_dump'

glove_training_data = util.get_glove_data(training_data_file)
util.dump_object(glove_training_data, glove_training_data_dump)

glove_test_data = util.get_glove_data(test_data_file)
util.dump_object(glove_test_data, glove_test_data_dump)