from setuptools import setup, find_packages

setup(
    name='sklearn_utils',
    version='0.0.14',
    packages=find_packages(),
    description='Sklearn utils',
    author="Muhammed Hasan Celik",
    author_email="hasancelik@std.sehir.edu.tr",
    url="https://github.com/MuhammedHasan/sklearn_utils",
    install_requires=[
        'numpy',
        'pyfunctional',
        'pandas',
        'scipy',
        'scikit-learn',
        'statsmodels',
        'seaborn'
    ],
    include_package_data=True,
    test_suite='sklearn_utils.tests',
    keywords=['scikit-learn', 'machine-learning', 'utility-library'],
    classifiers=[
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ])
