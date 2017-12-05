from sklearn.base import TransformerMixin
from sklearn_utils.utils import map_dict_list


class FeatureRenaming(TransformerMixin):
    '''
    Preprocessing to re-name features.
    '''

    def __init__(self, names):
        '''
        :names: dict which contain old feature names as key and new names as value.  
        '''
        self.names = names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        '''
        :X: list of dict
        '''
        return map_dict_list(
            X,
            key_func=lambda k, v: self.names[k],
            if_func=lambda k, v: k in self.names)
