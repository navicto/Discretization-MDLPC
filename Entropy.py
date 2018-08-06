from __future__ import division

__author__ = 'Victor Ruiz, vmr11@pitt.edu'

import pandas as pd
import numpy as np
from math import log

def entropy_numpy(data_classes, base=2):
    '''
    Computes the entropy of a set of labels (class instantiations)
    :param base: logarithm base for computation
    :param data_classes: Series with labels of examples in a dataset
    :return: value of entropy
    '''
    classes = np.unique(data_classes)
    N = len(data_classes)
    ent = 0  # initialize entropy

    # iterate over classes
    for c in classes:
        partition = data_classes[data_classes == c]  # data with class = c
        proportion = len(partition) / N
        #update entropy
        ent -= proportion * log(proportion, base)

    return ent

def cut_point_information_gain_numpy(X, y, cut_point):
    '''
    Return de information gain obtained by splitting a numeric attribute in two according to cut_point
    :param dataset: pandas dataframe with a column for attribute values and a column for class
    :param cut_point: threshold at which to partition the numeric attribute
    :param feature_label: column label of the numeric attribute values in data
    :param class_label: column label of the array of instance classes
    :return: information gain of partition obtained by threshold cut_point
    '''
    entropy_full = entropy_numpy(y)  # compute entropy of full dataset (w/o split)

    #split data at cut_point
    data_left_mask = X <= cut_point #dataset[dataset[feature_label] <= cut_point]
    data_right_mask = X > cut_point #dataset[dataset[feature_label] > cut_point]
    (N, N_left, N_right) = (len(X), data_left_mask.sum(), data_right_mask.sum())

    gain = entropy_full - (N_left / N) * entropy_numpy(y[data_left_mask]) - \
        (N_right / N) * entropy_numpy(y[data_right_mask])

    return gain