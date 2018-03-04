import unittest
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from .dynamic_pipeline import DynamicPipeline
from .feature_merger import FeatureMerger
from .feature_renaming import FeatureRenaming
from .fold_change_preprocessing import FoldChangeScaler
from .functional_enrichment_analysis import FunctionalEnrichmentAnalysis
from .standard_scale_by_label import StandardScalerByLabel
from .dict_input import DictInput


class TestDictInput(unittest.TestCase):
    def setUp(self):
        self.data = [
            {'a': 0, 'b': 2, 'c': 0, 'd': 3},
            {'a': 0, 'b': 1, 'c': 4, 'd': 3}
        ]

    def test_fit_transform(self):
        scaler = DictInput(StandardScaler())
        expected_data = [{'b': 1.0, 'c': -1.0}, {'b': -1.0, 'c': 1.0}]
        self.assertEqual(expected_data, scaler.fit_transform(self.data))

    def test_fit_transform_with_feature_selection(self):
        vt = DictInput(VarianceThreshold(), feature_selection=True)
        expected_data = [{'b': 2.0}, {'b': 1.0, 'c': 4.0}]
        self.assertEqual(expected_data, vt.fit_transform(self.data))

    def test_clone_safety(self):
        X = [self.data[0]] * 10 + [self.data[1]] * 10
        y = ['x'] * 10 + ['h'] * 10
        pipe = Pipeline([
            ('vt', DictInput(VarianceThreshold(), feature_selection=True)),
            ('vect', DictVectorizer()),
            ('clf', LogisticRegression())
        ])
        cross_val_score(pipe, X, y, cv=5, scoring='f1_micro')


class TestFoldChangeScaler(unittest.TestCase):
    def setUp(self):
        self.X = [{'a': 2.5, 'b': 5, 'c': 10},
                  {'a': 20, 'b': 40, 'c': 80, 'x': 1}]
        self.h = {'a': 10, 'b': 10, 'c': 10}
        self.X.append(self.h)
        self.y = ['b', 'b', 'h']
        self.scaler = FoldChangeScaler(reference_label='h')

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
        self.X = [{'a': 1, 'b': 2}, {'a': 3, 'b': 1}]

        class MyPipeline(DynamicPipeline):
            steps = {
                'vect': DictVectorizer(sparse=False),
                'std': StandardScaler(),
                'vect-std': DictInput(StandardScaler()),
            }

            default_steps = ['vect']

        self.cls = MyPipeline

    def test_new(self):
        pipe = self.cls()
        self.assertEqual(pipe.fit_transform(self.X).tolist(), [[1, 2], [3, 1]])

        pipe = self.cls(['vect'])
        self.assertEqual(pipe.fit_transform(self.X).tolist(), [[1, 2], [3, 1]])

        expected = [[-1, 1], [1, -1]]
        pipe = self.cls(['vect', 'std'])
        self.assertEqual(pipe.fit_transform(self.X).tolist(), expected)

        expected = [{'a': -1, 'b': 1}, {'a': 1, 'b': -1}]
        pipe = self.cls(['vect-std'])
        self.assertEqual(pipe.fit_transform(self.X), expected)


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


class TestFeatureMerger(unittest.TestCase):
    def setUp(self):
        self.features = {'A': ['a', 'b'], 'C': ['x', 'y']}
        self.X = [{'a': 1, 'b': 5}]

    def test_fit_transform(self):
        X_t = FeatureMerger(self.features, 'mean').fit_transform(self.X)
        self.assertEqual(X_t, [{'A': 3}])

        X_t = FeatureMerger(self.features, 'sum').fit_transform(self.X)
        self.assertEqual(X_t, [{'A': 6}])
