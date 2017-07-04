import pandas as pd
import numpy as np
import pytest

from pandas.util import testing as pdt
from numpy import testing as npt

from data import *
from plot import *

def test_input_required_decorator():
    @input_requires(['user_id'])
    def f(df):
        return df
    a = pd.DataFrame({'a': [1,2,3]})
    with pytest.raises(ValueError):
        f(a)

def test_check_output_schema():
    a = pd.DataFrame({'a': [1,2,3]})
    with pytest.raises(ValueError):
        check_output_schema(a, 2)

def test_check_nonconstant_col():
    @nonconstant_col
    def f(df):
        return df
    a = pd.DataFrame({'a': [1,1,1]})
    with pytest.raises(ValueError):
        f(a)

def test_check_unique_id():
    a = pd.DataFrame({'a': [1,1,2]})
    with pytest.raises(ValueError):
        check_unique_id(a, 'a')

def test_filter_week_window():
    df = pd.DataFrame({'start': ['2017-06-26']*6,
                       'date': ['2017-06-26','2017-07-03','2017-07-10','2017-07-17',
                                '2017-07-24','2017-07-31']})
    a = filter_week_window(df, 1, 3).astype(str).reset_index(drop=True)
    b = pd.DataFrame({'start': ['2017-06-26']*2, 'date': ['2017-07-03','2017-07-10'],
                      'filter_start': ['2017-07-03']*2, 'filter_end': ['2017-07-17']*2})
    b = b[['date', 'start', 'filter_start', 'filter_end']]
    pdt.assert_frame_equal(a, b)

def test_crosstab():
    df = pd.DataFrame({0: ['a','b','c','d','e']*2, 1: ['aa', 'bb']*5})
    crosstab(df, 0, 1)

def test_winsorize():
    df = pd.DataFrame({0: ['a','b','c','d','e']*2, 1: range(1, 11)})
    a = winsorize(df[1], .05).values
    b = pd.Series(range(2, 10)).values
    npt.assert_array_equal(a, b)

def test_top_n_cat():
    df = pd.DataFrame({0: ['a','a','a','b','b','d','e','f','g','h']})
    a = top_n_cat(df[0], 2)
    b = pd.Series(['a','a','a','b','b','other','other','other','other','other'], name=0)
    pdt.assert_series_equal(a, b)

def test_bin_cont():
    df = pd.DataFrame({'col': range(10)})
    a = bin_cont(df['col'], 5)
    assert a.unique().shape[0] == 2

    df = pd.DataFrame([1,3]*5 + [5,9]*10 + [12,17,11]*5 + [21,19,23]*5)
    a = bin_cont(df[0], 5)
    assert a.unique().shape[0] == 5

def test_treat():
    df = pd.DataFrame([1,2,3,4,5])
    a = treat(df[0], 5).values
    b = pd.Series([1,2,3,4,5]).values
    npt.assert_array_equal(a, b)

def test_mark_nth_week():
    df = pd.DataFrame({'start': ['2017-06-26']*6,
                       'date': ['2017-06-26','2017-07-03','2017-07-10','2017-07-17',
                                '2017-07-24','2017-07-31']})
    df['start'] = pd.to_datetime(df['start'])
    df['date'] = pd.to_datetime(df['date'])

    a = mark_nth_week(df)

    b = pd.DataFrame({'start': ['2017-06-26']*6,
                      'date': ['2017-06-26','2017-07-03','2017-07-10','2017-07-17',
                               '2017-07-24','2017-07-31'],
                      'nth_week': range(1,7)})
    b['start'] = pd.to_datetime(b['start'])
    b['date'] = pd.to_datetime(b['date'])
    b = b[['date', 'start', 'nth_week']]

    pdt.assert_frame_equal(a, b)

def test_time_diff():
    df = pd.DataFrame({'date': ['2017-06-26','2017-07-03','2017-07-10','2017-07-17',
                                '2017-07-24','2017-07-31']})
    df['date'] = pd.to_datetime(df['date'])
    a = time_diff(df, None)
    b = pd.DataFrame({'date': ['2017-06-26','2017-07-03','2017-07-10','2017-07-17',
                      '2017-07-24','2017-07-31'], 'time_diff': [np.NaN] + [604800.0]*5})
    b['date'] = pd.to_datetime(b['date'])
    pdt.assert_frame_equal(a, b)

def test_time_diff_users():
    df = pd.DataFrame({'user_id': [1,2,1,2,1,2], 'date': ['2017-06-26','2017-07-03',
                       '2017-07-10','2017-07-17', '2017-07-24','2017-07-31']})
    a = time_diff(df).reset_index(drop=True)
    b = pd.DataFrame({'user_id': [1,1,1,2,2,2], 'date': ['2017-06-26','2017-07-10',
                      '2017-07-24','2017-07-03', '2017-07-17','2017-07-31'],
                      'time_diff': [np.NaN, 1209600.0, 1209600.0]*2})
    b['date'] = pd.to_datetime(b['date'])
    b = b[['date', 'user_id', 'time_diff']]
    pdt.assert_frame_equal(a, b)

def test_mark_consec_run_same_user():
    df = pd.DataFrame({'user_id': [1,1,1,1,1], 'b': [1,2,3,1,2]})
    a = mark_consecutive_runs(df, 'b')

    b = pd.DataFrame({'user_id': [1,1,1,1,1], 'b': [1,2,3,1,2], 'run': [1,1,1,2,2]})
    b = b[['b', 'user_id', 'run']]
    pdt.assert_frame_equal(a, b)

def test_mark_consec_run_diff_user():
    df = pd.DataFrame({'user_id': [1,1,1,1,2,2,2], 'b': [1,2,10,3,4,2,1]})
    a = mark_consecutive_runs(df, 'b')

    b = pd.DataFrame({'user_id': [1,1,1,1,2,2,2], 'b': [1,2,10,3,4,2,1], 'run': [1,1,2,3,4,5,6]})
    b = b[['b', 'user_id', 'run']]
    pdt.assert_frame_equal(a, b)

def test_mark_consec_run_no_user():
    df = pd.DataFrame({'a': [1,1,1,1,1], 'b': [1,2,3,1,2]})
    a = mark_consecutive_runs(df, 'b', None)

    b = pd.DataFrame({'a': [1,1,1,1,1], 'b': [1,2,3,1,2], 'run': [1,1,1,2,2]})
    b = b[['a', 'b', 'run']]
    pdt.assert_frame_equal(a, b)

def test_mark_adjacent():
    df = pd.DataFrame({'user_id': [1,1,1,1,2,2,2,3], 'b': [1,1,1,2,2,2,4,5]})
    a = mark_adjacent_groups(df, 'b')

    b = pd.DataFrame({'user_id': [1,1,1,1,2,2,2,3], 'b': [1,1,1,2,2,2,4,5],
                      'group': [1,1,1,2,3,3,4,5]})
    b = b[['b', 'user_id', 'group']]
    pdt.assert_frame_equal(a, b)

def test_mark_adjacent_no_user():
    df = pd.DataFrame({'a': [1,1,1,1,2,2,2,3], 'b': [1,1,1,2,2,2,4,5]})
    a = mark_adjacent_groups(df, 'b', None)

    b = pd.DataFrame({'a': [1,1,1,1,2,2,2,3], 'b': [1,1,1,2,2,2,4,5],
                      'group': [1,1,1,2,2,2,3,4]})
    b = b[['a', 'b', 'group']]
    pdt.assert_frame_equal(a, b)

def test_dummies():
    df = pd.DataFrame({'user_id': range(9), 'cat': ['a', 'b', 'c']*3})
    a = dummies(df, 'cat').astype(int)
    b = pd.DataFrame({'user_id': range(9), 'a': [1,0,0]*3, 'b': [0,1,0]*3, 'c': [0,0,1]*3})
    b = b[['user_id', 'a', 'b', 'c']]
    pdt.assert_frame_equal(a, b)

def test_count_categorical():
    df = pd.DataFrame({'user_id': [0, 1, 2]*3, 'cat': ['a', 'b', 'c']*3})
    a = count_categorical(df, 'cat').astype(int)
    b = pd.DataFrame({'user_id': [0,1,2], 'a': [3,0,0], 'b': [0,3,0], 'c': [0,0,3]})
    b = b[['user_id', 'a', 'b', 'c']]
    pdt.assert_frame_equal(a, b)

def test_total_count():
    df = pd.DataFrame({'user_id': [1,1,2,2,2], 'col': range(10,60,10)})
    a = df.groupby('user_id')['col'].count().reset_index()
    b = pd.DataFrame({'user_id': [1, 2], 'col': [2, 3]})
    b = b[['user_id', 'col']]
    pdt.assert_frame_equal(a, b)

def test_total_value():
    df = pd.DataFrame({'user_id': [1,1,2,2,2], 'col': range(10,60,10)})
    a = df.groupby('user_id')['col'].sum().reset_index()
    b = pd.DataFrame({'user_id': [1, 2], 'col': [30, 120]})
    b = b[['user_id', 'col']]
    pdt.assert_frame_equal(a, b)

def test_count_categorical():
    df = pd.DataFrame({'user_id': [1,1,2,2,2], 'col': ['a', 'b', 'c', 'a', 'b']})
    a = df.pipe(dummies, 'col').groupby('user_id').sum().reset_index().astype(int)
    b = pd.DataFrame({'user_id': [1, 2], 'a': [1,1], 'b': [1,1], 'c': [0,1]})
    b = b[['user_id', 'a', 'b', 'c']]
    pdt.assert_frame_equal(a, b)

def test_plot_bar_single_col():
    df = pd.DataFrame({0: ['a', 'a', 'b', 'b', 'b', 'b', 'c', 'd', 'e', 'a']})
    a = plot_bar(df, 0).values
    b = pd.Series([.4,.3,.1,.1,.1]).values
    npt.assert_array_equal(a, b)

def test_plot_bar_multi_col():
    df = pd.DataFrame({'cat': ['a','a','b','a','c','d'], 'col1': [1,2,3,4,5,6],
                       'col2': [7,8,9,10,11,12]})
    a = plot_bar(df, 'cat', ['col1','col2'])

    b = pd.DataFrame({'a':[ 7/3., 25/3.], 'b': [3., 9.], 'c': [5., 11.],
                      'd': [6., 12.]}, index=['col1', 'col2'])
    b.index.name = 'variable'
    b.columns.name = 'cat'
    pdt.assert_frame_equal(a, b)

def test_plot_line():
    df = pd.DataFrame({'cat1': ['a','a','b','a','c','d'],
                       'cat2': ['mo', 'tu', 'we', 'th', 'fr', 'sa'],
                       'col1': [1,2,3,4,5,6], 'col2': [7,8,9,10,11,12]})

    a = plot_line(df, 'cat1', 'col1').values
    b = pd.Series([7/3., 3., 5., 6.]).values
    npt.assert_array_equal(a, b)

def test_plot_heatmap():
    df = pd.DataFrame({'cat1': ['a','a','b','a','b','b'], 'cat2': ['aa','bb']*3,
                       'col1': [1,2,3,4,5,6], 'col2': [1, 2]*3})

    a = plot_heatmap(df, 'cat1', 'cat2', 'col1')

    b = pd.DataFrame({'aa': [1, 4], 'bb': [3, 6]}, index=['a', 'b'])
    b.index.name = 'cat1'
    b.columns.name = 'cat2'
    pdt.assert_frame_equal(a, b)

def test_plot_hist():
    df = pd.DataFrame({'col': np.random.normal(0, 1, 1000), 'cat': ['a']*500 + ['b']*500})
    plot_hist(df, 'col')
    plot_hist(df, 'cat', 'col')

def test_plot_box():
    df = pd.DataFrame({'col1': np.random.normal(0, 1, 1000),
                       'col2' : np.random.normal(5, 10, 1000),
                       'cat': ['a']*500 + ['b']*500})
    plot_box(df, 'cat', 'col1')
    plot_box(df, 'cat', ['col1', 'col2'])

def test_plot_scatter():
    df = pd.DataFrame({'col1': np.random.normal(0, 1, 1000),
                       'col2' : np.random.normal(5, 10, 1000),
                       'cat': ['a']*500 + ['b']*500})

    plot_scatter(df, 'col1', 'col2')
    plot_scatter(df, 'cat', 'col1', 'col2')
    # very strange bug occurs when the above line is commented out

def test_ts_line():
    df = pd.DataFrame({'date': ['2017-01-01']*3 + ['2017-02-01']*2 + ['2017-03-01']*5,
                  'col': range(10, 110, 10), 'cat': ['a']*5 + ['b']*3 + ['c']*2,
                  'col2': np.random.normal(5,4,10)})
    df['date'] = pd.to_datetime(df['date'])
    print df.info()

    plot_ts_line(df)
    plot_ts_line(df, cat='cat')
    plot_ts_line(df, col='col')
    plot_ts_line(df, 'cat', 'col')

def test_ts_area():
    df = pd.DataFrame({'date': ['2017-01-01']*3 + ['2017-02-01']*2 + ['2017-03-01']*5,
                  'col': range(10, 110, 10), 'cat': ['a']*5 + ['b']*3 + ['c']*2,
                  'col2': np.random.normal(5,4,10)})
    df['date'] = pd.to_datetime(df['date'])

    plot_ts_area(df)
    plot_ts_area(df, cat='cat')
    plot_ts_area(df, col='col')
    plot_ts_area(df, 'cat', 'col')

def test_ts_bar():
    df = pd.DataFrame({'date': ['2017-01-01']*3 + ['2017-02-01']*2 + ['2017-03-01']*5,
                  'col': range(10, 110, 10), 'cat': ['a']*5 + ['b']*3 + ['c']*2,
                  'col2': np.random.normal(5,4,10)})
    df['date'] = pd.to_datetime(df['date'])

    plot_ts_bar(df)
    plot_ts_bar(df, cat='cat')
    plot_ts_bar(df, col='col')
    plot_ts_bar(df, 'cat', 'col')

def test_ts_box():
    df = pd.DataFrame({'date': ['2017-01-01']*3 + ['2017-02-01']*2 + ['2017-03-01']*5,
                  'col': range(10, 110, 10), 'cat': ['a']*5 + ['b']*3 + ['c']*2,
                  'col2': np.random.normal(5,4,10)})
    df['date'] = pd.to_datetime(df['date'])

    plot_ts_box(df, 'col')
