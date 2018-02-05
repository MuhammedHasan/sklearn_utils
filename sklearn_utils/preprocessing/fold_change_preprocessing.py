from sklearn.base import TransformerMixin
from sklearn_utils.utils import average_by_label, map_dict_list


class FoldChangeScaler(TransformerMixin):
    '''
    Scales by measured value by distance to mean according to time of value.
    Useful when you want to standart scale but no varience.
    '''

    def __init__(self, reference_label, bounds=(-10, 10)):
        """
        :reference_label: the label scaling will be performed by.
        :bounds: min-max values the fold change scaler can get.
        There are bounds because the scaling can provide unstable results.
        """
        self._min = bounds[0]
        self._max = bounds[1]
        self.reference_label = reference_label

    def fit(self, X, y):
        '''
        :X: list of dict
        :y: labels
        '''
        self._avgs = average_by_label(X, y, self.reference_label)
        return self

    def transform(self, X):
        return map_dict_list(
            X,
            value_func=self._scale,
            if_func=lambda k, v: k in self._avgs
        )

    def _scale(self, k, v):
        e = v / self._avgs[k]
        if self._avgs[k] > v:
            return max(1 - e**-1, self._min)
        return min(e - 1, self._max)
