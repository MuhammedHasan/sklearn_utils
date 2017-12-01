from functional import seq
from sklearn.base import TransformerMixin
from scipy.stats import fisher_exact

from sklearn_utils.utils import average_by_label


class FunctionalEnrichmentAnalysis(TransformerMixin):
    """Functional Enrichment Analysis"""

    def __init__(self,
                 reference_label,
                 feature_groups,
                 method="fisher_exact",
                 alternative='two-sided',
                 filter_func=None):
        '''
        :reference_label: label of refence values in the calculation
        :method: only fisher exact test avaliable so far
        :feature_groups: list of dict where keys are new feature and values are list of old features
        :filter_func: function return true or false
        '''
        if method != "fisher_exact":
            raise NotImplemented('Only fisher exact test is implemented')

        self.reference_label = reference_label
        self.feature_groups = feature_groups
        self.alternative = alternative
        self.filter_func = filter_func or (lambda x: round(x, 3) <= 0)

    def fit(self, X, y):
        self._references = average_by_label(X, y, self.reference_label)
        return self

    def transform(self, X, y=None):
        '''
        :X: list of dict
        :y: labels
        '''
        return [{
            new_feature: self._fisher_pval(x, old_features)
            for new_feature, old_features in self.feature_groups.items()
            if len(set(x.keys()) & set(old_features))
        } for x in X]

    def _filtered_values(self, x: dict, feature_set: list=None):
        '''
        :x: dict which contains feature names and values
        :return: pairs of values which shows number of feature makes filter function true or false
        '''
        feature_set = feature_set or x
        n = sum(self.filter_func(x[i]) for i in feature_set if i in x)
        return [len(feature_set) - n, n]

    def _contingency_table(self, x: dict, feature_set: list):
        return list(
            zip(*[
                self._filtered_values(xs, feature_set)
                for xs in [self._references, x]
            ]))

    def _fisher_pval(self, x: dict, feature_set: list):
        return fisher_exact(
            self._contingency_table(x, feature_set),
            alternative=self.alternative)[1]
