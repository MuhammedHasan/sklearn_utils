import math
import json
import gzip
import pickle

import pandas as pd

from .data_utils import map_dict_list


class SkUtilsIO:
    '''
    IO class to read and write dataset to file.
    '''

    def __init__(self, path, gz=False):
        '''
        :filename: file name with path.
        '''
        self.path = path
        self.gz = gz

    def from_csv(self, label_column='labels'):
        '''
        Read dataset from csv.
        '''
        df = pd.read_csv(self.path, header=0)
        X = df.loc[:, df.columns != label_column].to_dict('records')
        X = map_dict_list(X, if_func=lambda k, v: v and math.isfinite(v))
        y = list(df[label_column].values)
        return X, y

    def to_csv(self, X, y):
        '''
        Writes dataset to csv.
        '''
        # TODO: implement this
        raise NotImplementedError

    def from_json(self):
        '''
        Reads dataset from json.
        '''
        with gzip.open('%s.gz' % self.path,
                       'rt') if self.gz else open(self.path) as file:
            return list(map(list, zip(*json.load(file))))[::-1]

    def to_json(self, X, y):
        '''
        Reads dataset to csv.

        :param X: dataset as list of dict.
        :param y: labels.
        '''
        with gzip.open('%s.gz' % self.path, 'wt') if self.gz else open(
                self.path, 'w') as file:
            json.dump(list(zip(y, X)), file)

    def from_pickle(self):
        '''
        Reads dataset to pickle.
        '''
        raise NotImplementedError

    def to_pickle(self, X, y):
        '''
        Writes dataset to pickle.

        :param X: dataset as list of dict.
        :param y: labels.
        '''
        raise NotImplementedError
