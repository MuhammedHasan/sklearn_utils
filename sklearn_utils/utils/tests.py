import unittest

from .data_utils import *


class TestDataUtils(unittest.TestCase):
    def setUp(self):
        self.y = ['bc', 'bc', 'h', 'h']
        self.X = [{
            'a': 1,
            'b': 2
        }, {
            'a': 2,
            'b': 3
        }, {
            'a': 4,
            'b': 5
        }, {
            'a': 6,
            'b': 7
        }]

    def test_filter_by_label(self):
        X, y = filter_by_label(range(4), self.y, 'h')
        self.assertEqual(X, (2, 3))
        self.assertEqual(y, ('h', 'h'))

    def test_average_by_label(self):
        X_t = average_by_label(self.X, self.y, 'h')
        self.assertEqual(X_t, {'a': 5, 'b': 6})

    def test_map_dict(self):
        key_func = lambda k, v: k + '_'
        inc_func = lambda k, v: v + 1
        d_t = map_dict(self.X[0], key_func=key_func, value_func=inc_func)
        self.assertEqual(d_t, {'a_': 2, 'b_': 3})

    def test_map_dict_list(self):
        key_func = lambda k, v: k + '_'
        inc_func = lambda k, v: v + 1
        d_t = map_dict_list(self.X[:2], key_func=key_func, value_func=inc_func)
        self.assertEqual(d_t, [{'a_': 2, 'b_': 3}, {'a_': 3, 'b_': 4}])
