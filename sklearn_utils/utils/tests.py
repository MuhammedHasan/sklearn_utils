import unittest
from unittest import mock

import pandas as pd

from .data_utils import *
from .skutils_io import *


class TestDataUtils(unittest.TestCase):
    def setUp(self):
        self.y = ['bc', 'bc', 'h', 'h']
        self.X = [
            {'a': 1, 'b': 2},
            {'a': 2, 'b': 3},
            {'a': 4, 'b': 5},
            {'a': 6, 'b': 7}
        ]

    def _key_func(self, k, v):
        return k + '_'

    def _inc_func(self, k, v):
        return v + 1

    def test_filter_by_label(self):
        X, y = filter_by_label(range(4), self.y, 'h')
        self.assertEqual(X, (2, 3))
        self.assertEqual(y, ('h', 'h'))

    def test_average_by_label(self):
        X_t = average_by_label(self.X, self.y, 'h')
        self.assertEqual(X_t, {'a': 5, 'b': 6})

    def test_map_dict(self):
        d_t = map_dict(self.X[0],
                       key_func=self._key_func,
                       value_func=self._inc_func)
        self.assertEqual(d_t, {'a_': 2, 'b_': 3})

    def test_map_dict_list(self):
        d_t = map_dict_list(self.X[:2],
                            key_func=self._key_func,
                            value_func=self._inc_func)
        self.assertEqual(d_t, [{'a_': 2, 'b_': 3}, {'a_': 3, 'b_': 4}])

    def test_check_reference_label(self):
        check_reference_label(self.y, 'h')
        with self.assertRaises(ValueError):
            check_reference_label(self.y, 'x')

    def test_variance_threshold_on_df(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [1, 1, 1]})
        df_expected = pd.DataFrame({'a': [1, 2, 3]})
        return pd.testing.assert_frame_equal(
            variance_threshold_on_df(df), df_expected)

    @unittest.skip("issue in travis")
    def test_feature_importance_report(self):
        X = [
            {'a': 1, 'b': 2},
            {'a': 1, 'b': 2},
            {'a': 2, 'b': 2},
            {'a': 2, 'b': 2}
        ]
        df = feature_importance_report(X, self.y)
        self.assertListEqual(list(df.values[0]), [1, 2, float('inf'), 0])


class TestSkUtilsIO(unittest.TestCase):
    def setUp(self):
        self.X = [[1, 2]] * 4
        self.y = ['bc', 'bc', 'h', 'h']
        self.X_y = list(zip(self.y, self.X))
        self.io = SkUtilsIO('a')
        self.io_gz = SkUtilsIO('a', gz=True)
        self.df = pd.DataFrame(
            [[i, *j] for i, j in self.X_y],
            columns=['labels', 'x', 'y'])

    @mock.patch('gzip.open')
    @mock.patch('builtins.open')
    @mock.patch('json.dump')
    def test_to_json(self, mock_json_dump, mock_open, mock_gzip_open):

        self.io.to_json(self.X, self.y)
        mock_json_dump.assert_called_with(
            self.X_y, mock_open.return_value.__enter__.return_value)

        self.io_gz.to_json(self.X, self.y)
        mock_json_dump.assert_called_with(
            self.X_y, mock_gzip_open.return_value.__enter__.return_value)

    @mock.patch('gzip.open')
    @mock.patch('builtins.open')
    @mock.patch('json.load')
    def test_from_json(self, mock_json_load, mock_open, mock_gzip_open):
        mock_json_load.return_value = self.X_y
        self.assertEqual(self.io.from_json(), [self.X, self.y])
        self.assertEqual(self.io_gz.from_json(), [self.X, self.y])

    @mock.patch('pandas.read_csv')
    def test_from_csv(self, pd_read_csv):
        pd_read_csv.return_value = self.df
        X, y = self.io.from_csv()

        self.assertEqual(X, [{'x': i, 'y': j} for i, j in self.X])
        self.assertEqual(y, self.y)

        self.df.loc[3, 'y'] = float('nan')
        pd_read_csv.return_value = self.df
        X, y = self.io.from_csv()
        self.assertEqual(X[-1], {'x': 1})

        self.df.loc[3, 'y'] = ''
        pd_read_csv.return_value = self.df
        X, y = self.io.from_csv()
        self.assertEqual(X[-1], {'x': 1})
