from collections import defaultdict

import pandas as pd


def filter_by_label(X, y, label, reverse=False):
    """Select items with label from dataset"""
    return list(
        zip(*filter(lambda t: (not reverse) == (t[1] == label), zip(X, y))))


def average_by_label(X, y, label):
    """returns average dictinary from list of dictionary for give label"""
    return defaultdict(float,
                       pd.DataFrame.from_records(
                           filter_by_label(X, y, label)[0]).mean().to_dict())


def map_dict(d, key_func=None, value_func=None):
    '''
    :d: dict
    :key_func: func which will run on key.
    :value_func: func which will run on values.
    '''
    key_func = key_func or (lambda k, v: k)
    value_func = value_func or (lambda k, v: v)
    return {key_func(*k_v): value_func(*k_v) for k_v in d.items()}


def map_dict_list(ds, key_func=None, value_func=None):
    '''
    :ds: list of dict
    :key_func: func which will run on key.
    :value_func: func which will run on values.
    '''
    return [map_dict(d, key_func, value_func) for d in ds]
