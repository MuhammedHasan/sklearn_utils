from collections import defaultdict

import pandas as pd
from statsmodels.sandbox.stats.multicomp import multipletests
from sklearn.feature_selection import VarianceThreshold, f_classif


def filter_by_label(X, y, ref_label, reverse=False):
    '''
    Select items with label from dataset.

    :param X: dataset
    :param y: labels
    :param ref_label: reference label
    :param bool reverse: if false selects ref_labels else eliminates
    '''
    check_reference_label(y, ref_label)

    return list(zip(*filter(lambda t: (not reverse) == (t[1] == ref_label),
                            zip(X, y))))


def average_by_label(X, y, ref_label):
    '''
    Calculates average dictinary from list of dictionary for give label

    :param List[Dict] X: dataset
    :param list y: labels
    :param ref_label: reference label
    '''
    # TODO: consider to delete defaultdict
    return defaultdict(float,
                       pd.DataFrame.from_records(
                           filter_by_label(X, y, ref_label)[0]
                       ).mean().to_dict())


def map_dict(d, key_func=None, value_func=None, if_func=None):
    '''
    :param dict d: dictionary
    :param func key_func: func which will run on key.
    :param func value_func: func which will run on values.
    '''
    key_func = key_func or (lambda k, v: k)
    value_func = value_func or (lambda k, v: v)
    if_func = if_func or (lambda k, v: True)
    return {
        key_func(*k_v): value_func(*k_v)
        for k_v in d.items() if if_func(*k_v)
    }


def map_dict_list(ds, key_func=None, value_func=None, if_func=None):
    '''
    :param List[Dict] ds: list of dict
    :param func key_func: func which will run on key.
    :param func value_func: func which will run on values.
    '''
    return [map_dict(d, key_func, value_func, if_func) for d in ds]


def check_reference_label(y, ref_label):
    '''
    :param list y: label
    :param ref_label: reference label
    '''
    set_y = set(y)
    if ref_label not in set_y:
        raise ValueError('There is not reference label in dataset. '
                         "Reference label: '%s' "
                         'Labels in dataset: %s' % (ref_label, set_y))


def variance_threshold_on_df(df: pd.DataFrame, threshold=0):
    vt = VarianceThreshold(threshold)
    vt.fit(df.values)
    return df.iloc[:, vt.variances_ > threshold]


def feature_importance_report(X,
                              y,
                              threshold=0.001,
                              correcting_multiple_hypotesis=True,
                              method='fdr_bh',
                              alpha=0.1,
                              sort_by='pval'):
    '''
    Provide signifance for features in dataset with anova using multiple hypostesis testing

    :param X: List of dict with key as feature names and values as features
    :param y: Labels
    :param threshold: Low-variens threshold to eliminate low varience features
    :param correcting_multiple_hypotesis: corrects p-val with multiple hypotesis testing
    :param method: method of multiple hypotesis testing
    :param alpha: alpha of multiple hypotesis testing
    :param sort_by: sorts output dataframe by pval or F
    :return: DataFrame with F and pval for each feature with their average values 
    '''
    df = variance_threshold_on_df(
        pd.DataFrame.from_records(X), threshold=threshold)

    F, pvals = f_classif(df.values, y)

    if correcting_multiple_hypotesis:
        _, pvals, _, _ = multipletests(pvals, alpha=alpha, method=method)

    df['labels'] = y
    df_mean = df.groupby('labels').mean().T

    df_mean['F'] = F
    df_mean['pval'] = pvals

    return df_mean.sort_values(sort_by, ascending=True)
