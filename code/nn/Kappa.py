# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 18:55:38 2016

@author: Pranav
"""

import skll.metrics as metric

# Calculates the average quadratic kappa for the entire essay set
def get_average_kappa(targets, predictions):
	num = len(targets)
	total_kappa = 0

	for i in xrange(0, num):
		total_kappa += metric.kappa([targets[i][0]], [predictions[i][0]], 'quadratic')

	avg_kappa  = float(total_kappa) / float(num)
	return avg_kappa
