import itertools
import numpy as np
import pandas as pd

from datetime import datetime
from pandas.tseries.offsets import *
from patsy import dmatrices

import re

def top_n_cat(a, n=5):
    a = a.fillna('missing')
    counts = a.value_counts()
    top = counts.iloc[:n].index
    return a.apply(lambda x: x if x in top else 'other')

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

@input_requires(['date', 'start'])
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

def mark_consecutive_runs(df, col, reset_count_on='user_id'):
    df = df.copy()
    is_nonconsecutive_number = df[col] != df[col].shift() + 1

    if reset_count_on:
        is_diff_user = df['user_id'] != df['user_id'].shift()
        df['run'] = (is_nonconsecutive_number | is_diff_user).cumsum()
    else:
        df['run'] = (is_nonconsecutive_number).cumsum()
    return df

@input_requires(['date', 'start'])
def mark_nth_week(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['start'] = pd.to_datetime(df['start'])
    df['nth_week'] = (df['date'] - df['start']).dt.days / 7 + 1
    df['nth_week'] = df['nth_week'].astype(int)
    df.loc[df['nth_week'] < 0, 'nth_week'] = 0
    return df

#@input_requires(['date'])
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

# #@input_requires(['user_id'])
# def dummies(df, col, obs_unit, top=None):
#     df = df.copy()
#
#     if top:
#         df[col] = top_n_cat(df[col], top)
#
#     dummy_col = pd.get_dummies(df[col])
#     dummy_col.columns = [str(i) for i in dummy_col.columns]
#
#     df = pd.concat([df[[obs_unit]], dummy_col], axis=1)
#     return df

#####
# dataframe functions
def concat(df, df_list, **kwargs):
    dfs = [df.reset_index(drop=True)] + [pd.DataFrame(df_i).reset_index(drop=True) for df_i in df_list]
    return pd.concat(dfs, **kwargs)

def crosstab(df, col1, col2, col3=None, aggfunc=np.mean, **kwargs):
    if col3 is None:
        return pd.crosstab(df[col1], df[col2], **kwargs)
    else:
        return pd.crosstab(df[col1], df[col2], df[col3], aggfunc=aggfunc, **kwargs)

def merge(df, df_list, on, how, **kwargs):
    df = df.copy()

    for df_i in df_list:
        if not df.pipe(is_unique, on) and not df_i.pipe(is_unique, on):
            raise Exception, 'Many-to-many join will result in duplicate rows.'
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
    new_cols = df.pipe(formula, 'col1*col2 + col3/col4')
    df.pipe(concat, new_cols, axis=1)
    '''
    X = dmatrix(formula, df)
    X = pd.DataFrame(X)
    return X

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

# miscellaneous functions
def disjoint_sliding_window(x, n=2):
    '''
    [0,1,2,3] -> [(0,1), (2,3)]
    '''
    return zip(x, x[1:])[::n]
#####

# def load(filename, date_cols, folder='/Users/alexhuang/Documents/data/gobble_data/'):
#     return pd.read_csv(folder + filename, parse_dates=date_cols)

def contains_any(a, strings):
    return any([x for x in strings if x in a])

def contains_none(a, strings):
    return bool(1 - contains_any(a, strings))

def get_cols(df, include=None, exclude=None, return_df=True):
    df = df.copy()

    if include:
        include = set(include)
    else:
        include = set(df.columns)

    if exclude:
        exclude = set(exclude)
    else:
        exclude = set()

    c = [i for i in df.columns if contains_any(i, include) and contains_none(i, exclude)]

    if return_df:
        return df[c]
    else:
        return c

# def add_col(df, col, name=None):
#     col = pd.DataFrame(col).reset_index(drop=True)
#     if name:
#         col.columns = [name]
#     return pd.concat([df.reset_index(drop=True), col], axis=1)

# [(0,2), (2,4), (4,6)]
def intervals(start, end, step=2):
    return zip(range(start,end+step,step), range(start+step,end+step,step))

def mark_nth_day(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['start'] = pd.to_datetime(df['start'])
    df['day'] = (df['date'] - df['start']).dt.days + 1
    df['day'] = df['day'].astype(int)
    df.loc[df['day'] < 0, 'day'] = 0
    return df

def mark_within_hour(df, date_col):
    df = df.copy()
    df = df.sort_values(by=['anonymous_id', date_col])
    is_diff_number = (df[date_col] - df[date_col].shift()).dt.total_seconds()/3600 >= 1
    is_diff_user = df['anonymous_id'] != df['anonymous_id'].shift()
    df['group'] = (is_diff_number | is_diff_user).cumsum()
    return df

def consecutive_runs(df):
    df = df.copy()
    df = df.pipe(mark_nth_week)
    df = df[~df['nth_week'].isnull()]
    df = df.pipe(mark_consecutive_runs, 'nth_week')
    df = df.groupby(['user_id', 'run']).size().reset_index().drop('run', axis=1)
    return df

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

def split_list_col(df, col):
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(df[col].values.tolist())],
        axis=1).drop(col, 1)

#?
def merge_unique(df1, df2, on, how):
    return df1.reset_index().merge(df2, on=on, how=how).groupby('index').head(1).drop('index', 1)

def group_into_list(df, group, col):
    return df.groupby(group)[col].apply(list).reset_index()
