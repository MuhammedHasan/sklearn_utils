from setuptools import setup, find_packages

setup(
    name='sklearn_utils',
    version='0.0.1',
    packages=find_packages(),
    author="Muhammed Hasan Celik",
    author_email="hasancelik@std.sehir.edu.tr",
    install_requires=['scikit-learn'],
    include_package_data=True,
    test_suite='sklearn_utils.tests',
    classifiers=[
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ])
