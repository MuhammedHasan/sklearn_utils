import unittest

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold

from .inverse_dict_vectorizer import InverseDictVectorizer
from .basic_fold_change_preprocessing import BasicFoldChangeScaler


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


class TestBasicFoldChangeScaler(unittest.TestCase):
    def setUp(self):
        self.X = [{'a': 2.5, 'b': 5, 'c': 10}, {'a': 20, 'b': 40, 'c': 80}]
        self.h = {'a': 10, 'b': 10, 'c': 10}
        self.X.append(self.h)
        self.y = ['b', 'b', 'h']
        self.scaler = BasicFoldChangeScaler()

    def test_fit(self):
        self.scaler.fit(self.X, self.y)
        self.assertEqual(self.scaler._avgs, self.h)

    def test_transform(self):
        X = self.scaler.fit_transform(self.X, self.y)
        expected_X = [{
            'a': -3,
            'b': -1,
            'c': 0
        }, {
            'a': 1,
            'b': 3,
            'c': 7
        }, {
            'a': 0,
            'b': 0,
            'c': 0
        }]
        self.assertEqual(X, expected_X)

    def test_scale(self):
        self.scaler._avgs = {'a': 10**-6}
        self.assertEqual(self.scaler._scale('a', 10**6), 10)

        self.scaler._avgs = {'a': 2}
        self.assertEqual(self.scaler._scale('a', 8), 3)

        self.scaler._avgs = {'a': 10**6}
        self.assertEqual(self.scaler._scale('a', 10**-6), -10)
