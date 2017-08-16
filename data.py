from datetime import datetime
from pandas.tseries.offsets import *
from patsy import dmatrix

import numpy as np
import pandas as pd

def input_requires(cols):
    def decorator(f):
        def wrapper(df, *args, **kwargs):
            for i in cols:
                if i not in df.columns:
                    raise ValueError('df needs column named %s' % i)
            return f(df, *args, **kwargs)
        return wrapper
    return decorator

def check_output_schema(df, num_cols):
    if len(df.columns) != num_cols:
        raise ValueError, 'Incorrect number of columns'
    return df

def output_schema(num_cols):
    def decorator(f):
        def wrapper(df, *args, **kwargs):
            output = f(df, *args, **kwargs)
            if len(output.columns) != num_cols:
                raise ValueError, 'Incorrect number of columns'
            return output
        return wrapper
    return decorator

def nonconstant_col(f):
    def wrapper(df, *args, **kwargs):
        output = f(df, *args, **kwargs)
        if len(output.iloc[:, -1].value_counts()) == 1:
            raise ValueError, 'Contains constant column'
        return output
    return wrapper

def check_unique_id(df, id_col):
    if len(df.groupby(id_col).size().value_counts()) > 1:
        raise ValueError, 'Contains duplicate %s' % id_col
    return df

def top_n_cat(a, n=5):
    a = a.fillna('missing')
    counts = a.value_counts()
    top = counts.iloc[:n].index
    return a.apply(lambda x: x if x in top else 'other')

def filter_week_window(df, n1, n2):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['start'] = pd.to_datetime(df['start'])
    df['filter_start'] = df['start'] + DateOffset(weeks=n1)
    df['filter_end'] = df['start'] + DateOffset(weeks=n2)
    df = df.query('filter_start <= date < filter_end')
    return df

def mark_adjacent_groups(df, col, reset_count_on='user_id'):
    df = df.copy()
    is_diff_number = df[col] != df[col].shift()

    if reset_count_on:
        is_diff_user = df['user_id'] != df['user_id'].shift()
        df['group'] = (is_diff_number | is_diff_user).cumsum()
    else:
        df['group'] = (is_diff_number).cumsum()

    return df

# def is_diff_element(x):
#     '''
#     df[col].apply(is_diff_element)
#     df.pipe(transform, {'col': 'is_diff_element'})
#     '''
#     return x != x.shift()
#
# def is_nonconsecutive(x):
#     return x != x.shift() + 1
#
# def is_within_hour(x):
#     return (x - x.shift()).dt.total_seconds() / 3600 >= 1

def time_diff(df, date_col='date', groups=None):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if groups is None:
        df = df.sort_values(by=date_col)
        df['time_diff'] = df[date_col].diff().dt.total_seconds()
    else:
        df = df.sort_values(by=groups + [date_col])
        df['time_diff'] = df[date_col].diff().dt.total_seconds()
        for g in groups:
            df.loc[df[g] != df[g].shift(1), 'time_diff'] = np.nan

    return df

def mark_nth_week(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['start'] = pd.to_datetime(df['start'])
    df['nth_week'] = (df['date'] - df['start']).dt.days / 7 + 1
    df['nth_week'] = df['nth_week'].astype(int)
    df.loc[df['nth_week'] < 0, 'nth_week'] = 0
    return df

def mark_nth_day(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['start'] = pd.to_datetime(df['start'])
    df['day'] = (df['date'] - df['start']).dt.days + 1
    df['day'] = df['day'].astype(int)
    df.loc[df['day'] < 0, 'day'] = 0
    return df

def dummies(df, col, obs_unit, top=None):
    df = df.copy()

    if top:
        df[col] = top_n_cat(df[col], top)

    dummy_col = pd.get_dummies(df[col])
    dummy_col.columns = [str(i) for i in dummy_col.columns]

    df = pd.concat([df[[obs_unit]], dummy_col], axis=1)
    return df

def add_col(df, col, name=None):
    col = pd.DataFrame(col).reset_index(drop=True)
    if name:
        col.columns = [name]
    return pd.concat([df.reset_index(drop=True), col], axis=1)

def intervals(start, end, step=2):
    return zip(range(start,end+step,step), range(start+step,end+step,step))

def get_ts_counts(df, start, end, name):
    a = mark_nth_day(df).groupby(['user_id', 'day']).size().unstack()
    missing_days = np.setdiff1d(np.array(range(end)), a.columns)
    a = a.reindex(columns=np.append(a.columns.values, missing_days)).sort_index(1)
    a = a.iloc[:, start:end]
    a.columns = ["day_%s_%s" % (i, name) for i in a.columns]
    return a.reset_index()

def get_ts_sum(df, start, end, col, name):
    a = mark_nth_day(df).groupby(['user_id', 'day'])[col].sum().unstack()
    missing_days = np.setdiff1d(np.array(range(end)), a.columns)
    a = a.reindex(columns=np.append(a.columns.values, missing_days)).sort_index(1)
    a = a.iloc[:, start:end]
    a.columns = ["day_%s_%s" % (i, name) for i in a.columns]
    return a.reset_index()

# def parse_tree(s, X):
#     tokens = [i for i in re.split(r'([,()])', s) if i != '']
#     function_dict = {'add': '+', 'sub': '-', 'log': 'np.log', 'min': 'min', 'div': '/'}
#
#     parsed = ''
#     current_op = None
#     for i in tokens:
#         i = i.strip()
#         if i in function_dict:
#             if i in ['add', 'sub', 'div']:
#                 current_op = i
#             else:
#                 parsed += i
#         elif i == '(':
#             parsed += i
#         elif i == ',':
#             if current_op:
#                 parsed += ' %s ' % function_dict[current_op]
#                 current_op = None
#             else:
#                 parsed += i + ' '
#         else:
#             if i[0] == 'X':
#                 col = X.columns[int(i[1:])]
#                 parsed += "%s" % col
#             else:
#                 parsed += i
#
#     return parsed

#####
# general dataframe functions
def concat(df, df_list, **kwargs):
    dfs = [df.reset_index(drop=True)] + [pd.DataFrame(df_i).reset_index(drop=True) for df_i in df_list]
    return pd.concat(dfs, **kwargs)

# have way to reduce number of categories using top_cat?
def crosstab(df, col1, col2, col3=None, aggfunc=np.mean, **kwargs):
    if col3 is None:
        return pd.crosstab(df[col1], df[col2], **kwargs)
    else:
        return pd.crosstab(df[col1], df[col2], df[col3], aggfunc=aggfunc, **kwargs)

def merge(df, df_list, on, how, **kwargs):
    df = df.copy()

    for df_i in df_list:
        if not df.pipe(is_unique, on) and not df_i.pipe(is_unique, on):
            raise Exception, 'Many-to-many join results in duplicate rows.'
        df = df.merge(df_i, on, how, **kwargs)

    return df

def is_unique(df, col):
    if len(df.groupby(col).size().value_counts()) > 1:
        return False
    else:
        return True

def duplicates(df, col):
    counts = df.groupby(col).size()
    dups = counts[counts > 1].index
    return df[df[col].isin(dups)].sort_values(by=col)

def query(df, func):
    '''
    df.pipe(query, lambda x: x['col'] > 5)
    '''
	return df[func(df)]

def quantile(df, col, q=10):
    df = df.copy()
    df = df.sort_values(by=col, ascending=False).reset_index(drop=True)
    df['%s quantile' % col] = pd.qcut(df.index, q, labels=False) + 1
    return df

def transform(df, cf_dict, append=False):
    df = df.copy()

    for cols, func in cf_dict.items():
        col_names = list(cols)
        if append:
            df = df.pipe(concat, list(df[col_names].apply(func)), axis=1)
        else:
            df[col_names] = df[col_names].apply(func)

    return df

def formula(df, formula):
    '''
    new_cols = df.pipe(formula, 'col1*col2 + col3/col4 + col5**2 + np.log(col6 + 1)')
    df.pipe(concat, new_cols, axis=1)
    '''
    X = dmatrix(formula, df)
    return pd.DataFrame(X)

# general functions for transactional data
# all functions need start and date columns
def time_window(df, offset1, offset2, freq='W'):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['start'] = pd.to_datetime(df['start'])
    df['filter_start'] = df.set_index('start').shift(periods=offset1, freq=freq).index
    df['filter_end'] = df.set_index('start').shift(periods=offset2, freq=freq).index
    df = df.query('filter_start <= date < filter_end')
    return df

def mark_timestep(df, unit):
    '''
    df.pipe(mark_timestep, 'week')
    '''
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['start'] = pd.to_datetime(df['start'])

    time_difference = (df['date'] - df['start']).dt.total_seconds()
    unit_dict = {'sec': 1, 'min': 60, 'hour': 3600, 'day': 3600*24, 'week': 3600*24*7}

    df['timestep'] = time_difference / unit_dict[unit] + 1
    df['timestep'] = df['timestep'].astype(int)

    return df

# needs user id
def timeseries(df, start, end, unit):
    a = mark_timestep(df, unit).groupby(['user_id', 'day']).size().unstack()
    missing_days = np.setdiff1d(np.array(range(end)), a.columns)
    a = a.reindex(columns=np.append(a.columns.values, missing_days)).sort_index(1)
    a = a.iloc[:, start:end]
    return a.reset_index()

# general series functions
def onehot(x):
    '''
    df[col].apply(onehot)
    df.pipe(transform, {'col': onehot})
    '''
    return pd.get_dummies(x)

def inv_onehot(df):
    '''
    df[cols].apply(inv_onehot)
    df.pipe(transform, {('val1', 'val2', 'val3'): inv_onehot})
    '''
    return df.apply(lambda x: x.idxmax(), axis=1)

def binarize(df):
    '''
    df[col].apply(binarize)
    df.pipe(transform, {'col': binarize})
    '''
    return df.apply(lambda x: 1 if x > 0 else 0, axis=1)

def props(x):
    '''
    df[col].apply(props)
    df.pipe(transform, {'col': props})
    '''
    return x/float(sum(x))

def missing_ind(x):
    '''
    df[col].apply(missing_ind)
    df.pipe(transform, {'col': missing_ind}, append=True)
    '''
    return x.apply(lambda x: 1 if pd.isnull(x) else 0)

def top_cat(x, n=5):
    '''
    df[col].apply(top_cat)
    df.pipe(transform, {'col': top_cat}, append=True)
    '''
    counts = x.fillna('missing').value_counts()
    top = counts.iloc[:n].index
    return x.apply(lambda x: x if x in top else 'other')

# miscellaneous functions
def disjoint_sliding_window(x, n=2):
    '''
    disjoint_sliding_window([0,1,2,3], 2) -> [(0,1), (2,3)]
    '''
    return zip(x, x[1:])[::n]

def disjoint_intervals(start, end, step=2):
    '''
    disjoint_intervals(0, 6, 2) -> [(0,2), (2,4), (4,6)]
    disjoint_intervals(0, 6, 3) -> [(0,3), (3,6)]
    '''
    return zip(range(start, end+step, step), range(start+step, end+step, step))

def contains_any(s, str_list):
    return any([i for i in str_list if i in s])
