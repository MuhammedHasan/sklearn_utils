import unittest

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold

from .inverse_dict_vectorizer import InverseDictVectorizer


class TestInverseDictVectorizer(unittest.TestCase):
    def setUp(self):
        self.data = [{
            'a': 0,
            'b': 2,
            'c': 0,
            'd': 3
        }, {
            'a': 0,
            'b': 1,
            'c': 4,
            'd': 3
        }]
        self.vect = DictVectorizer(sparse=False)
        self.trans_data = self.vect.fit_transform(self.data)

    def test_fit_transform(self):
        scaler = InverseDictVectorizer(self.vect)
        expected_data = [{'b': 2, 'd': 3}, {'b': 1, 'd': 3, 'c': 4}]
        self.assertEqual(expected_data, scaler.transform(self.trans_data))

    def test_fit_transform_with_feature_selection(self):
        vt = VarianceThreshold()
        data = vt.fit_transform(self.trans_data)
        scaler = InverseDictVectorizer(self.vect, vt)
        expected_data = [{'b': 2, 'c': 0}, {'b': 1, 'c': 4}]
        self.assertEqual(expected_data, scaler.fit_transform(data))
