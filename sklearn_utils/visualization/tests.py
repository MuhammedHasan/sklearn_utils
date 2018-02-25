import unittest

from .heatmap import plot_heatmap


class TestVisualization(unittest.TestCase):

    def setUp(self):
        self.y = ['bc', 'bc', 'h', 'h']
        self.X = [
            {'a': 1, 'b': 2},
            {'a': 1, 'b': 3},
            {'a': 2, 'b': 5},
            {'a': 2, 'b': 7}
        ]

    @unittest.skip("issue in unittest")
    def test_heatmap(self):
        plot_heatmap(self.X, self.y)
