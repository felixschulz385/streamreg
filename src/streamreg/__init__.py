"""
StreamReg: Streaming Regression for Large Datasets

Main API:
- OLS: Ordinary least squares estimator
- TwoSLS: Two-stage least squares estimator
"""

from streamreg.api import OLS, TwoSLS
from streamreg.data import StreamData, DatasetInfo
from streamreg.results import RegressionResults
from streamreg.formula import FormulaParser

__all__ = [
    'OLS',
    'TwoSLS',
    'StreamData',
    'DatasetInfo',
    'RegressionResults',
    'FormulaParser'
]

__version__ = '0.1.0'
