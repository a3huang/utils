import pandas as pd
import unittest

from pandas.util import testing as pdt
from numpy import testing as npt

from plot import winsorize, _top_n_cat

def test_winsorize():
    df = pd.DataFrame({0: ['a','b','c','d','e']*2, 1: range(1, 11)})
    a = winsorize(df[1], .05).values
    b = pd.Series(range(2, 10)).values
    npt.assert_array_equal(a, b)

def test_top_n_cat():
    df = pd.DataFrame({0: ['a','a','a','b','b','d','e','f','g','h']})
    a = _top_n_cat(df[0], 2)
    b = pd.Series(['a','a','a','b','b','other','other','other','other','other'], name=0)
    pdt.assert_series_equal(a, b)
