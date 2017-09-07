from datetime import datetime
from pandas.tseries.offsets import *
from patsy import dmatrix

import numpy as np
import pandas as pd

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

    ex) disjoint_intervals(0, 6, 2) -> [(0, 2), (2, 4), (4, 6)]
    ex) disjoint_intervals(0, 6, 3) -> [(0, 3), (3, 6)]
    '''

    return zip(range(start, end+step, step), range(start+step, end+step, step))

# def disjoint_sliding_windows(x, n=2):
#     '''
#     Generate disjoint sliding windows of a given width across a list of integers.
#
#     ex) disjoint_sliding_windows([0, 1, 2, 3], 2) -> [(0, 1), (2, 3)]
#     '''
#
#     return zip(x, x[1:])[::n]

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

def qcut(a, q=10):
    '''
    Cut a series into quantiles with the smallest quantile equal to 1.

    ex) df[col].pipe(qcut)
    '''

    a = a.sort_values().reset_index().reset_index()
    a['quantile'] = pd.qcut(a['level_0'], 10, labels=False) + 1
    return a.set_index('index').sort_index().reset_index()['quantile']

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

def rates(a):
    '''
    Calculate proportions by dividing a series by its sum.

    ex) df[col].value_counts().pipe(rates)
    '''

    return x / float(sum(x))

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

def slice(df, f):
    '''
    Slice a dataframe using complex boolean expressions without having to
    specify its name. Useful when method chaining.

    ex) df.pipe(slice, lambda x: x['date'] > '2017-01-01')
    '''

    return df[f(df)]

def rename(df, name_list):
    '''
    Rename the columns of a dataframe without having to respecify old names.

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
    Show rows containing duplicate values of a given column.

    ex) df.pipe(show_duplicates, col)
    '''

    counts = df.groupby(col).size()
    duplicates = counts[counts > 1].index
    return df[df[col].isin(duplicates)].sort_values(by=col)

def filter_time_window(df, left_offset, right_offset, frequency):
    '''
    Filter rows of a transactional dataframe with date lying within the specified time window.

    ex) df.pipe(filter_time_window, 1, 2, freq='7D')
    '''

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['start'] = pd.to_datetime(df['start'])
    df['left_bound'] = df.set_index('start').shift(periods=left_offset, freq=freq).index
    df['right_bound'] = df.set_index('start').shift(periods=right_offset, freq=freq).index
    df = df.query('left_bound <= date < right_bound')
    return df
#####

# def mark_timestep(df, unit):
#     '''
#     df.pipe(mark_timestep, 'week')
#     '''
#     df = df.copy()
#     df['date'] = pd.to_datetime(df['date'])
#     df['start'] = pd.to_datetime(df['start'])
#
#     time_difference = (df['date'] - df['start']).dt.total_seconds()
#     unit_dict = {'sec': 1, 'min': 60, 'hour': 3600, 'day': 3600*24, 'week': 3600*24*7}
#
#     df['timestep'] = time_difference / unit_dict[unit] + 1
#     df['timestep'] = df['timestep'].astype(int)
#
#     return df

def timeseries(df, start, end, datecol, freq, aggfunc):
    a = df.set_index(datecol).to_period(freq).reset_index()\
            .groupby(['user_id', datecol]).agg(aggfunc).unstack()
    missing_days = np.setdiff1d(np.array(range(end)), a.columns)
    a = a.reindex(columns=np.append(a.columns.values, missing_days)).sort_index(1)
    a = a.iloc[:, start:end]
    return a.reset_index()

def timeseries_new(df, start, end, datecol, freq, aggfunc):
    a = df.set_index(datecol).to_period(freq).reset_index()\
        .groupby(['user_id', datecol]).agg(aggfunc).unstack()
    min_date = df[datecol].min() + DateOffset(days=-6)
    max_date = df[datecol].max()
    date_range = pd.date_range(start=min_date, end=max_date, freq='W-MON').astype(str).values
    missing_days = np.setdiff1d(daterange, a.columns)
    a = a.reindex(columns=np.append(a.columns.values, missing_days)).sort_index(1)
    a = a.iloc[:, start:end]
    return a.reset_index()

def get_feature_scores(col_names, scores, top=5):
    '''
    ex) get_feature_scores(X_train.columns, model.feature_importances_)
    '''
    return pd.DataFrame(sorted(zip(col_names, scores), key=lambda x: x[1], reverse=True)[:top])

def interaction(df, col1, col2):
    '''
    Create interaction terms between 2 variables.

    ex) df.pipe(interaction, col1, col2)
    '''

    formula = '%s:%s - 1' % (col1, col2)
    X = dmatrix(formula, df)
    return pd.DataFrame(X, columns=X.design_info.column_names)
