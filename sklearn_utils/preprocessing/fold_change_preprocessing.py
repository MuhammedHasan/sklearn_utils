from sklearn.base import TransformerMixin
from sklearn_utils.utils import average_by_label, map_dict_list


class FoldChangeScaler(TransformerMixin):
    '''
    Scales by measured value by distance to mean according to time of value.
    Useful when you want to standart scale but no varience.
    '''

    def __init__(self, bounds=(-10, 10)):
        self._min = bounds[0]
        self._max = bounds[1]

    def fit(self, X, y):
        self._avgs = average_by_label(X, y, 'h')
        return self

    def transform(self, X):
        return map_dict_list(X, value_func=self._scale)

    def _scale(self, k, v):
        e = v / self._avgs[k]
        if self._avgs[k] > v:
            return max(1 - e**-1, self._min)
        return min(e - 1, self._max)
