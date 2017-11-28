import unittest

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold

from .inverse_dict_vectorizer import InverseDictVectorizer
from .fold_change_preprocessing import FoldChangeScaler
from .feature_renaming import FeatureRenaming
from .dynamic_preprocessing import DynamicPreprocessing
from .functional_enrichment_analysis import FunctionalEnrichmentAnalysis
from .standard_scale_by_label import StandardScalerByLabel


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


class TestFoldChangeScaler(unittest.TestCase):
    def setUp(self):
        self.X = [{'a': 2.5, 'b': 5, 'c': 10}, {'a': 20, 'b': 40, 'c': 80}]
        self.h = {'a': 10, 'b': 10, 'c': 10}
        self.X.append(self.h)
        self.y = ['b', 'b', 'h']
        self.scaler = FoldChangeScaler()

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


class TestFeatureRenaming(unittest.TestCase):
    def setUp(self):
        self.preprocessing = FeatureRenaming({'x': 'y'})

    def test_transform(self):
        self.assertEqual(self.preprocessing.transform([{'x': 1}]), [{'y': 1}])


class TestDynamicPreprocessing(unittest.TestCase):
    def setUp(self):
        pass

    def test_new(self):
        pass


class TestFunctionalEnrichmentAnalysis(unittest.TestCase):
    def setUp(self):
        self.X = [{
            'GLUDym': 1,
            'GLUNm': -1,
            'GLNS': 0.0001,
            'METyLATthc': 100,
            'xxx': 1
        }, {
            'GLUDym': 1,
            'GLUNm': 2,
            'GLNS': -1
        }]
        self.y = ['h', 'x']

        self.groups = {
            'Glutamate metabolism': ['GLUDym', 'GLUNm', 'GLNS', 'METyLATthc']
        }

        self.preprocessing = FunctionalEnrichmentAnalysis('h', self.groups)

    def test_filtered_values(self):
        values = self.preprocessing._filtered_values(self.X[0])
        self.assertEqual(values, [3, 2])

        values = self.preprocessing._filtered_values(
            self.X[0], ['GLUDym', 'GLUNm', 'GLNS'])
        self.assertEqual(values, [1, 2])

    def test_contingency_table(self):
        self.preprocessing._references = self.X[1]

        values = self.preprocessing._contingency_table(
            self.X[0], ['GLUDym', 'GLUNm', 'GLNS'])
        self.assertEqual(values, [(2, 1), (1, 2)])

    def test_fisher_pval(self):
        self.preprocessing._references = self.X[1]

        pval = self.preprocessing._fisher_pval(self.X[0],
                                               list(self.groups.values())[0])

        self.assertEqual(pval, 1)

    def test_fit(self):
        self.preprocessing.fit(self.X, self.y)
        self.assertEqual(dict(self.preprocessing._references), self.X[0])

    def test_transform(self):
        pvals = self.preprocessing.fit_transform(self.X, self.y)
        self.assertEqual(list(pvals[0].keys())[0], 'Glutamate metabolism')
        self.assertEqual(list(pvals[0].values())[0], 1)


class TestStandardScalerByLabel(unittest.TestCase):
    def setUp(self):
        self.scaler = StandardScalerByLabel('h')
        self.X = [[10], [10], [10], [0], [0], [0]]
        self.y = ['bc', 'bc', 'bc', 'h', 'h', 'h']

    def test_partial_fit(self):
        self.scaler.partial_fit(self.X, self.y)
        expected_X = self.X
        transformed_X = self.scaler.transform(self.X).tolist()
        self.assertEqual(expected_X, transformed_X)
