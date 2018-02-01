import unittest
from unittest import mock

from .data_utils import *
from .skutils_io import *


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
        def key_func(k, v): return k + '_'

        def inc_func(k, v): return v + 1
        d_t = map_dict(self.X[0], key_func=key_func, value_func=inc_func)
        self.assertEqual(d_t, {'a_': 2, 'b_': 3})

    def test_map_dict_list(self):
        def key_func(k, v): return k + '_'

        def inc_func(k, v): return v + 1
        d_t = map_dict_list(self.X[:2], key_func=key_func, value_func=inc_func)
        self.assertEqual(d_t, [{'a_': 2, 'b_': 3}, {'a_': 3, 'b_': 4}])

    def test_check_reference_label(self):
        check_reference_label(self.y, 'h')
        with self.assertRaises(ValueError):
            check_reference_label(self.y, 'x')


class TestSkUtilsIO(unittest.TestCase):
    def setUp(self):
        self.X = [[1]] * 4
        self.y = ['bc', 'bc', 'h', 'h']
        self.X_y = [
            ['bc', [1]],
            ['bc', [1]],
            ['h', [1]],
            ['h', [1]],
        ]
        self.io = SkUtilsIO('a')
        self.io_gz = SkUtilsIO('a', gz=True)

    @mock.patch('gzip.open')
    @mock.patch('builtins.open')
    @mock.patch('json.dump')
    def test_write_json(self, mock_json_dump, mock_open, mock_gzip_open):

        self.io.to_json(self.X, self.y)
        mock_json_dump.assert_called_with(
            self.X_y, mock_open.return_value.__enter__.return_value)

        self.io_gz.to_json(self.X, self.y)
        mock_json_dump.assert_called_with(
            self.X_y, mock_gzip_open.return_value.__enter__.return_value)

    @mock.patch('gzip.open')
    @mock.patch('builtins.open')
    @mock.patch('json.load')
    def test_write_json(self, mock_json_load, mock_open, mock_gzip_open):

        mock_json_load.return_value = self.X_y
        self.assertEqual(self.io.from_json(), [self.X, self.y])
        self.assertEqual(self.io_gz.from_json(), [self.X, self.y])

    @mock.patch('gzip.open')
    @mock.patch('builtins.open')
    @mock.patch('json.load')
    def test_write_json(self, mock_json_load, mock_open, mock_gzip_open):

        mock_json_load.return_value = self.X_y
        self.assertEqual(self.io.from_json(), [self.X, self.y])
        self.assertEqual(self.io_gz.from_json(), [self.X, self.y])
