import itertools
import numpy as np
import pandas as pd

from datetime import datetime
from pandas.tseries.offsets import *

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

@input_requires(['date'])
def time_diff(df, group='user_id'):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    if group is None:
        df = df.sort_values(by='date')
        df['time_diff'] = df['date'].diff().dt.total_seconds()
    else:
        df = df.sort_values(by=[group, 'date'])
        df['time_diff'] = df['date'].diff().dt.total_seconds()
        df.loc[df[group] != df[group].shift(1), 'time_diff'] = np.nan

    return df

@input_requires(['user_id'])
def dummies(df, col, top=None):
    df = df.copy()

    if top:
        df[col] = top_n_cat(df[col], top)

    dummy_col = pd.get_dummies(df[col])
    dummy_col.columns = [str(i) for i in dummy_col.columns]

    df = pd.concat([df[['user_id']], dummy_col], axis=1)
    return df

def crosstab(df, col1, col2, col3=None, aggfunc=np.mean, **kwargs):
    df = df.copy()

    if col3 is None:
        return pd.crosstab(df[col1], df[col2], **kwargs)
    else:
        return pd.crosstab(df[col1], df[col2], df[col3], aggfunc=aggfunc, **kwargs)
#######

def merge(df, df_list, on='user_id'):
    for i in df_list:
        df = df.merge(i, on=on, how='left')
    return df

def stack(df_list):
    for i, df in enumerate(df_list):
        df['group'] = i
    return pd.concat(df_list)

def missing_indicator(df, col):
    df = df.copy()
    df.loc[:, '%s_missing' % col] = df[col].apply(lambda x: 1 if pd.isnull(x) else 0)
    return df

def load(filename, date_cols, folder='/Users/alexhuang/Documents/data/gobble_data/'):
    return pd.read_csv(folder + filename, parse_dates=date_cols)

def col_in(df, col, values):
    return df[df[col].isin(values)]

def col_between(df, left, col, right):
    return df[(df[col] > left) & (df[col] < right)]

def get_dups(df, col):
    a = df.groupby(col).size()
    dup = a[a > 1].index
    return df[df[col].isin(dup)].sort_values(by=col)

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

def add_col(df, col, name=None):
    col = pd.DataFrame(col).reset_index(drop=True)
    if name:
        col.columns = [name]
    return pd.concat([df.reset_index(drop=True), col], axis=1)

def add_agg_col(df, group, func, col=None):
    df = df.copy()

    if col:
        df[func] = df.groupby(group)[col].transform(func)
    else:
        df['count'] = df.groupby(group).transform('count').iloc[:, -1]

    return df.groupby(group).head(1)

######

def top_n(df, col, n=5):
    df = df.copy()
    df[col] = df[col].fillna('missing')
    counts = df[col].value_counts()
    top = counts.iloc[:n].index
    df[col] = df[col].apply(lambda x: x if x in top else 'other')
    return df

def transform(df, trans_dict):
    df = df.copy()
    for cols, trans in trans_dict.items():
        col_names = list(cols)
        df[col_names] = df[col_names].apply(trans)
    return df

def pca_transform(df, n_comp=2):
    pca = make_pipeline(StandardScaler(), PCA())
    return pd.DataFrame(pca.fit_transform(X)[:, :n_comp], columns=['PCA %s' % i for i in range(1, n_comp+1)])

def categorize(df, cols, name):
    df = df.copy()
    df[name] = df[cols].apply(lambda x: x.idxmax(), axis=1)
    return df.drop(cols, 1)



# needs date filter_start, filter_end, start, end column
def frequency(df, group, col):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['date_string'] = df['date'].dt.date
    df['total'] = df.groupby(group)['date_string'].transform('count')
    df = df.groupby(group).head(1)

    try:
        df['weeks'] = (df['filter_end'] - df['filter_start']).dt.days / 7.0
    except:
        df['end'] = df['end'].fillna(datetime.now())
        df['weeks'] = (df['end'] - df['start']).dt.days / 7.0

    df['frequency'] = df['total'] / df['weeks']
    df = df[[group, 'frequency']]
    return df

# make id default col?
# needs user id column
def get_grouped_rates(df, group, col):
    # get counts aggregated by user id
    a = df.pipe(add_agg_col, 'user_id', col, 'count')
    return a[a[col] > 0].groupby(group)['count'].size() / a.groupby(group)['count'].size()
def get_rate(df, col):
    a = df[col].value_counts(dropna=False)
    return a / float(sum(a))

# add default "names" to all other functions as well
def add_dow_offset(df, date_col, name='next_dow', **kwargs):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df[name] = df[date_col].dt.to_period('W').dt.start_time + DateOffset(**kwargs)
    return df

def add_date_offset(df, date_col, name='next', **kwargs):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df[name] = df[date_col].dt.date + DateOffset(**kwargs)
    return df

def consecutive_runs(df):
    df = df.copy()
    df = df.pipe(mark_nth_week)
    df = df[~df['nth_week'].isnull()]
    df = df.pipe(mark_consecutive_runs, 'nth_week')
    df = df.groupby(['user_id', 'run']).size().reset_index().drop('run', axis=1)
    return df

def get_weekly_ts(df, window, name):
    a, b = window
    return df.pipe(filter_week_window, a, b)\
              .pipe(mark_nth_week)\
              .query('nth_week >= 0')\
              .pipe(dummies, 'nth_week')\
              .groupby('user_id').sum().reset_index()\
              .pipe(name_with_template, name)

# need error checking on preserve
def my_query(df, query, on='user_id', preserve='group'):
    return df[[on, preserve]].merge(df.query(query).pipe(remove, [preserve]), on=on, how='left')

def binarize(x):
    return x.apply(lambda x: 1 if x > 0 else 0)

def combine_ratio(df, cols, name):
    df = df.copy()
    c = get_cols(df, cols, return_df=False)
    c = disjoint_window(c, len(cols))

    # if isinstance(cols[0], (list, tuple)):
    for i, pair in enumerate(c):
        col_pair = list(pair)
        df["%s_%s" % (name, i)] = df[col_pair].apply(lambda x: 0 if x[0] == 0 else x[1]/x[0], axis=1)
        df = df.drop(col_pair, 1)
    # else:
    #     df[name] = df[cols].apply(lambda x: 0 if x[0] == 0 else x[1]/x[0], axis=1)
    #     df = df.drop(cols, 1)
    return df

def combine_prod(df, cols, name):
    df = df.copy()
    df[name] = df[cols].apply(lambda x: 0 if x[0] == 0 else x[1]*x[0], axis=1)
    df = df.drop(cols, 1)
    return df

# [(0,2), (2,4), (4,6)]
def intervals(start, end, step=2):
    return zip(range(start,end+step,step), range(start+step,end+step,step))

# [0,1,2,3] -> [(0,1), (2,3)]
def disjoint_window(a, n=2):
    return zip(a, a[1:])[::n]

def mark_nth_day(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['start'] = pd.to_datetime(df['start'])
    df['day'] = (df['date'] - df['start']).dt.days + 1
    df['day'] = df['day'].astype(int)
    df.loc[df['day'] < 0, 'day'] = 0
    return df

def get_ts_counts(df, n, name):
    a = mark_nth_day(df).groupby(['user_id', 'day'])['id'].nunique().unstack().iloc[:, :n].fillna(0)
    a.columns = ["day_%s_%s" % (i, name) for i in a.columns]
    return a

def parse_tree(s, X):
    tokens = [i for i in re.split(r'([,()])', s) if i != '']
    function_dict = {'add': '+', 'sub': '-', 'log': 'np.log', 'min': 'min', 'div': '/'}

    parsed = ''
    current_op = None
    for i in tokens:
        i = i.strip()
        if i in function_dict:
            if i in ['add', 'sub', 'div']:
                current_op = i
            else:
                parsed += i
        elif i == '(':
            parsed += i
        elif i == ',':
            if current_op:
                parsed += ' %s ' % function_dict[current_op]
                current_op = None
            else:
                parsed += i + ' '
        else:
            if i[0] == 'X':
                col = X.columns[int(i[1:])]
                parsed += "%s" % col
            else:
                parsed += i

    return parsed

def window_event_count(df, windows, name):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['start'] = pd.to_datetime(df['start'])

    l = []
    for a, b in windows:
        start = df['start'] + DateOffset(weeks=a)
        end = df['start'] + DateOffset(weeks=b)
        col = df[(df['date'] >= start) & (df['date'] < end)].groupby('user_id')['id'].count().reset_index()
        col.columns = ['user_id', '%s_%s/%s' % (name, a, b)]
        l.append(col)

    return df[['user_id']].pipe(merge, l).groupby('user_id').head(1).fillna(0)
