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
#####

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

def disjoint_intervals(start, end, step=2):
    '''
    Generate 2-tuples of integers with a given range and step. Typically used
    to represent disjoint time ranges.

    ex) disjoint_intervals(0, 6, 2) -> [(0,2), (2,4), (4,6)]
    ex) disjoint_intervals(0, 6, 3) -> [(0,3), (3,6)]
    '''

    return zip(range(start, end+step, step), range(start+step, end+step, step))

def disjoint_sliding_windows(x, n=2):
    '''
    Generate sliding windows of a given width across a list of integers.

    ex) disjoint_sliding_windows([0, 1, 2, 3], 2) -> [(0, 1), (2, 3)]
    '''

    return zip(x, x[1:])[::n]

def cut(a, bin_width=None, bin_range=None, num_bins=None):
    '''
    Cut a continuous variable into bins of desired widths or number.

    ex) df[col].pipe(cut, num_bins=10)
    ex) df[col].pipe(cut, bin_width=30, range=(0, 100))
    '''

    if bin_range is None:
        bin_range = a.min(), a.max()

    if bin_width is None:
        bin_width = int(np.ceil((bin_range[1] - bin_range[0]) / num_bins))

    elif num_bins is None:
        num_bins = int(np.ceil((bin_range[1] - bin_range[0]) / bin_width))

    else:
        raise Exception, 'Need to specify either one of bin_width or num_bins.'

    min_edge = np.floor(bin_range[0] / bin_width)
    bin_edges = [min_edge + bin_width * i for i in range(num_bins + 1)]
    return pd.cut(a, bins=bin_edges, include_lowest=True)

def top(a, n):
    '''
    Keep the n most common levels of a categorical variable and label the rest as 'other'.

    ex) df[cat].pipe(top, 5)
    ex) df.assign(cat=lambda x: top(x[cat], 5))
    '''

    counts = a.fillna('missing').value_counts()
    top = counts.iloc[:n].index
    return a.apply(lambda x: x if x in top else 'other')

def dummy(a):
    '''
    Turn a categorical variable into dummy indicator variables.

    ex) df[col].apply(dummy)
    ex) df.pipe(cbind, df[col].pipe(dummy))
    '''

    return pd.get_dummies(a)

def undummy(a):
    '''
    Turn dummy indicator variables back into a categorical variable.

    ex) df[cols].apply(undummy)
    ex) df.pipe(cbind, df.iloc[:, 5:10].pipe(undummy))
    '''

    return a.apply(lambda x: x.idxmax(), axis=1)

def reduce_cardinality(a, n):
    '''
    Reduce the number of unique values of a variable. For a categorical variable,
    n specifies the number of categories. For a continuous variable, n specifies
    the bin widths.

    ex) df[col].pipe(reduce_cardinality, num_values=5)
    '''

    if a.dtype == 'O':
        return top(a, n)

    elif a.dtype in ['int32', 'int64', 'float32', 'float64']:
        return cut(a, bin_width=n)

def cbind(df, obj, **kwargs):
    '''
    Append a column or dataframe to an existing dataframe as new columns.

    ex) df.pipe(cbind, a)
    '''

    objects = [df.reset_index(drop=True), pd.DataFrame(obj).reset_index(drop=True)]
    return pd.concat(objects, axis=1, **kwargs)

def table(df, row_var, col_var, val_var=None, row_n=None, col_n=None, agg_func=np.mean, **kwargs):
    '''
    Calculate the cross tabulation between 2 categorical variables.

    ex) df.pipe(table, cat1, cat2)
    ex) df.pipe(table, cat1, cat2, col)
    ex) df.pipe(table, col, cat, row=5).iloc[:5]
    '''

    df = df.copy()

    if row:
        df[row_var] = df[row_var].pipe(reduce_cardinality, row_n)
    if col:
        df[col_var] = df[col_var].pipe(reduce_cardinality, col_n)

    if val_var is None:
        return pd.crosstab(df[row_var], df[col_var], **kwargs)
    else:
        return pd.crosstab(df[row_var], df[col_var], df[val_var], aggfunc=agg_func, **kwargs)

def merge(df, df_list, on, how, **kwargs):
    '''
    Merge a list of dataframes with an existing dataframe.

    ex) df.pipe(merge, [df1, df2, df3], on='user_id', how='left')
    '''

    df = df.copy()

    for df_i in df_list:
        df = df.merge(df_i, on=on, how=how, **kwargs)

    return df
#####

def query(df, func):
    '''
    Query a dataframe using complex boolean expressions without having to
    specify its name. Useful in the middle of a long method chain.

    ex) df.pipe(query, lambda x: x['date'] > '2017-01-01')
    '''

    return df[func(df)]

def rename(df, name_list):
    '''
    Rename the columns of a dataframe without having to specify its old names.

    ex) df.pipe(rename, ['col1', 'col2', 'col3'])
    '''

    df = df.copy()
    df = pd.DataFrame(df, index=df.index)
    df.columns = name_list
    return df

def check_unique(df, col):
    '''
    Check if column values are unique.

    ex) df.pipe(check_unique, col)
    '''

    if len(df.groupby(col).size().value_counts()) > 1:
        return False
    else:
        return True

def show_duplicates(df, col):
    '''
    Show rows where column value is duplicated.

    ex) df.pipe(show_duplicates, col)
    '''

    counts = df.groupby(col).size()
    duplicates = counts[counts > 1].index
    return df[df[col].isin(duplicates)].sort_values(by=col)

def interaction(df, col1, col2):
    '''
    Create a column(s) for the interaction between 2 variables.

    ex) df.pipe(interaction, col1, col2)
    '''

    formula = '%s:%s - 1' % (col1, col2)
    X = dmatrix(formula, df)
    return pd.DataFrame(X, columns=X.design_info.column_names)

def rates(x):
    '''
    Normalize a series by dividing by its sum.

    ex) df[col].value_counts().pipe(rates)
    '''

    return x / float(sum(x))

def qbin(x, q=10):
    '''
    Calculate the quantiles of a series with the largest quantile being 1.

    ex) df[col].pipe(quantile)
    '''

    a = x.sort_values().reset_index().reset_index()
    a['quantile'] = pd.qcut(a['level_0'], 10, labels=False) + 1
    return a.set_index('index').sort_index().reset_index()['quantile']

#####
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

def missing_ind(x):
    '''
    df[col].apply(missing_ind)
    df.pipe(transform, {'col': missing_ind}, append=True)
    '''
    return x.apply(lambda x: 1 if pd.isnull(x) else 0)

def contains_any(s, str_list):
    return any([i for i in str_list if i in s])

def count_val(df, val):
    return df.value_counts().loc[val]

def seq_props(x):
    return x / x[0]

def drop_consec_dups(df, col):
    return df[df[col] != df[col].shift()]

def get_feature_scores(df, scores, top=5):
    return pd.DataFrame(sorted(zip(df.columns, scores), key=lambda x: x[1], reverse=True)[:top])

def grouped_rates(x, col):
    return x / x.groupby(col).sum()
