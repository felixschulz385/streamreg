"""
StreamReg: Streaming Regression for Large Datasets

Main API:
- OLS: Ordinary least squares estimator
- TwoSLS: Two-stage least squares estimator
"""

from streamreg.api import OLS, TwoSLS, ols, twosls
from streamreg.data import StreamData, DatasetInfo
from streamreg.results import RegressionResults
from streamreg.formula import FormulaParser

__all__ = [
    'OLS',
    'TwoSLS',
    'ols',
    'twosls',
    'StreamData',
    'DatasetInfo',
    'RegressionResults',
    'FormulaParser'
]

__version__ = '0.1.0'
