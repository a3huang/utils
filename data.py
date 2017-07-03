import itertools
import numpy as np
import pandas as pd

from datetime import datetime
from pandas.tseries.offsets import *

def _top_n_cat(a, n=5):
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
        if len(output.iloc[:, -1].value_counts()) == 0:
            raise ValueError, 'Contains constant column'
        return output
    return wrapper

def check_unique_id(df, id_col):
    if len(df.groupby(id_col).size().value_counts()) > 1:
        raise ValueError, 'Contains duplicate %s' % id_col
    return df

@input_requires(['date', 'start', 'boundary'])
def filter_before_boundary(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['start'] = pd.to_datetime(df['start'])
    df['boundary'] = pd.to_datetime(df['boundary'])
    df['filter_start'] = df['start']
    df['filter_end'] = df['boundary']
    df = df.query('filter_start <= date < filter_end')
    return df

@input_requires(['date', 'start'])
def filter_first_n_weeks(df, n):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['start'] = pd.to_datetime(df['start'])
    df['filter_start'] = df['start']
    df['filter_end'] = df['start'] + DateOffset(weeks=n)
    df = df.query('filter_start <= date < filter_end')
    return df

@input_requires(['date', 'start'])
def filter_last_n_weeks(df, n):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['start'] = pd.to_datetime(df['start'])
    df['filter_start'] = datetime.now() - DateOffset(months=n)
    df['filter_end'] = datetime.now()
    df = df.query('filter_start <= date < filter_end')
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

#
def mark_adjacent_groups(df, col, reset_count_on='user_id'):
    df = df.copy()
    is_diff_number = df[col] != df[col].shift()

    if reset_count_on:
        is_diff_user = df['user_id'] != df['user_id'].shift()
        df['group'] = (is_diff_number | is_diff_user).cumsum()
    else:
        df['group'] = (is_diff_number).cumsum()

    return df

#
def mark_consecutive_runs(df, col, reset_count_on='user_id'):
    df = df.copy()
    is_nonconsecutive_number = df[col] != df[col].shift() + 1

    if reset_count_on:
        is_diff_user = df['user_id'] != df['user_id'].shift()
        df['run'] = (is_nonconsecutive_number | is_diff_user).cumsum()
    else:
        df['run'] = (is_nonconsecutive_number).cumsum()
    return df

#
@input_requires(['date', 'start'])
def mark_nth_week(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['start'] = pd.to_datetime(df['start'])
    df['nth_week'] = (df['date'] - df['start']).dt.days / 7 + 1
    df['nth_week'] = df['nth_week'].astype(int)
    return df

#
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

def crosstab(df, col1, col2, col3=None, aggfunc=np.mean, **kwargs):
    df = df.copy()

    if col3 is None:
        return pd.crosstab(df[col1], df[col2], **kwargs)
    else:
        return pd.crosstab(df[col1], df[col2], df[col3], aggfunc=aggfunc, **kwargs)

#
@input_requires(['user_id'])
def dummies(df, col, top=None):
    df = df.copy()

    if top:
        df[col] = _top_n_cat(df[col], top)

    dummy_col = pd.get_dummies(df[col])
    dummy_col.columns = [str(i) for i in dummy_col.columns]

    df = pd.concat([df[['user_id']], dummy_col], axis=1)
    return df

def remove(df, cols):
    return df[df.columns.difference(cols)]

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

def contains_none_of(a, strings):
    return bool(1 - contains_any(a, strings))

def get_columns_with(df, include=None, exclude=None):
    df = df.copy()

    if include:
        include = set(include)
    else:
        include = set(df.columns)

    if exclude:
        exclude = set(exclude)
    else:
        exclude = set()

    c = [i for i in df.columns if contains_any(i, include) and contains_none_of(i, exclude)]

    return c

def add_column(df, col, name=None):
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
