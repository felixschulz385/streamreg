"""
StreamReg: Streaming Regression for Large Datasets

Main API:
- ols(): Fit OLS regression
- twosls(): Fit 2SLS/IV regression
"""

from gnt.analysis.streamreg.api import ols, twosls
from gnt.analysis.streamreg.data import StreamData, DatasetInfo
from gnt.analysis.streamreg.results import RegressionResults
from gnt.analysis.streamreg.formula import FormulaParser

__all__ = [
    'ols',
    'twosls',
    'StreamData',
    'DatasetInfo',
    'RegressionResults',
    'FormulaParser'
]

__version__ = '0.1.0'
