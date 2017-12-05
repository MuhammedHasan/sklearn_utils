from sklearn.base import TransformerMixin
from sklearn_utils.utils import map_dict_list, map_dict


class FeatureRenaming(TransformerMixin):
    '''
    Preprocessing to re-name features.
    '''

    def __init__(self, names, case_sensetive=False):
        '''
        :names: dict which contain old feature names as key and new names as value.  
        :case_insensetive: performs mactching case sensetive
        '''
        self.names = names if case_sensetive else map_dict(
            names, key_func=lambda k, v: k.lower())

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        '''
        :X: list of dict
        '''
        return map_dict_list(
            X,
            key_func=lambda k, v: self.names[k.lower()],
            if_func=lambda k, v: k.lower() in self.names)
