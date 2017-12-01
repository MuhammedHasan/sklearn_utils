# sklearn_utils

[![Build Status](https://travis-ci.org/MuhammedHasan/sklearn_utils.svg?branch=master)](https://travis-ci.org/MuhammedHasan/sklearn_utils) [![Documentation Status](https://readthedocs.org/projects/sklearn-utils/badge/?version=latest)](http://sklearn-utils.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/MuhammedHasan/sklearn_utils/branch/master/graph/badge.svg)](https://codecov.io/gh/MuhammedHasan/sklearn_utils)

Utility functions, preprocessing steps, and class I need during in my research and developement projects in scikit learn.

## Installation

You can install `sklearn-utils` with `pip`:

```
pip install sklearn-utils
```

## Examples

If you want to scale your data based on reference values you may use StandardScalerByLabel. For example, I scale all the blood sample by healthy samples. 

```
from sklearn_utils.preprocessing import StandardScalerByLabel

preprocessing = StandardScalerByLabel('healthy')
X_t = preprocessing.fit_transform(X, y)

```

Or you may want your list of dict in the end of sklearn pipeline, after set of operations and feature selection.

```
from sklearn_utils.preprocessing import InverseDictVectorizer

vect = DictVectorizer(sparse=False) 
skb = SelectKBest(k=100)
pipe = Pipeline([
    ('vect', vect),
    ('skb', skb),
    ('inv_vect', InverseDictVectorizer(vect, skb))
])

X_t = pipe.fit_transform(X, y)

```

For more features, You can check the documentation.

## Documentation

The documentation of the project avaiable in http://sklearn-utils.rtfd.io .



