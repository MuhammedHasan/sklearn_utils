import numpy as np
from sklearn.base import TransformerMixin


class FeatureMerger(TransformerMixin):
    """Merge some features based on given strategy."""

    def __init__(self, features, strategy='mean'):
        '''
        :features: dict which contain new feature as key and old features as list in values.
        :strategy: strategy to merge features. 'mean', 'sum' and lambda function accepted. 
        Lambda function accepts list of values as input.
        '''
        self.features = features
        if strategy == 'mean':
            self.strategy = np.mean
        elif strategy == 'sum':
            self.strategy = np.sum

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [self._transform(x) for x in X]

    def _transform(self, x):
        new_features = dict()
        
        for f, fs in self.features.items():
            features = [x[i] for i in fs if i in x]
            if features:
                new_features[f] = self.strategy(features)

        return new_features
