import copy
from sklearn.base import TransformerMixin, clone
from sklearn.feature_extraction import DictVectorizer


class DictInput(TransformerMixin):
    """Converts a preprocessing step to accept list of dict."""

    def __init__(self, transformer, feature_selection=False, sparse=False):
        '''
        :param transformer: Sklearn transformer
        :param feature_selection: is this transformer perform feature selection.
        '''
        self.transformer = transformer
        self.feature_selection = feature_selection
        self.sparse = sparse

    def fit(self, X, y=None):
        self.dict_vectorizer_ = DictVectorizer(sparse=self.sparse)
        self.transformer.fit(self.dict_vectorizer_.fit_transform(X, y), y)

        if self.feature_selection:
            names = self.transformer.get_support()
            self.clone_dict_vectorizer_ = copy.deepcopy(self.dict_vectorizer_)
            self.clone_dict_vectorizer_.restrict(names)

        return self

    def transform(self, X):
        '''
        :param X: features.
        '''
        inverser_tranformer = self.dict_vectorizer_
        if self.feature_selection:
            inverser_tranformer = self.clone_dict_vectorizer_

        return inverser_tranformer.inverse_transform(
            self.transformer.transform(
                self.dict_vectorizer_.transform(X)))


class DfInput(DictInput):

    def __init__(self, transformer, feature_selection=False, sparse=False):
        # TODO: implement this.
        raise NotImplementedError()
