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

@input_requires(['date', 'start', 'boundary'])
def filter_before_boundary(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['filter_start'] = df['start']
    df['filter_end'] = df['boundary']
    df = df.query('filter_start <= date < filter_end')
    return df

@input_requires(['date', 'start'])
def filter_first_n_weeks(df, n):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['filter_start'] = df['start']
    df['filter_end'] = df['start'] + DateOffset(weeks=n)
    df = df.query('filter_start <= date < filter_end')
    return df

@input_requires(['date', 'start'])
def filter_last_n_weeks(df, n):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['filter_start'] = datetime.now() - DateOffset(months=n)
    df['filter_end'] = datetime.now()
    df = df.query('filter_start <= date < filter_end')
    return df

@input_requires(['date', 'start'])
def filter_week_window(df, n1, n2):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['filter_start'] = df['start'] + DateOffset(weeks=n1)
    df['filter_end'] = df['start'] + DateOffset(weeks=n2)
    df = df.query('filter_start <= date < filter_end')
    return df

#
@input_requires(['user_id'])
def mark_adjacent_groups(df, col):
    df = df.copy()
    is_diff_number = df[col] != df[col].shift()
    is_diff_user = df['user_id'] != df['user_id'].shift()
    df['group'] = (is_diff_number | is_diff_user).cumsum()
    return df

#
@input_requires(['user_id'])
def mark_consecutive_runs(df, col):
    df = df.copy()
    is_nonconsecutive_number = df[col] != df[col].shift() + 1
    is_diff_user = df['user_id'] != df['user_id'].shift()
    df['run'] = (is_nonconsecutive_number | is_diff_user).cumsum()
    return df

#
@input_requires(['date', 'start'])
def mark_nth_week(df):
    df = df.copy()
    df['nth_week'] = (df['date'] - df['start']).dt.days / 7 + 1
    df['nth_week'] = df['nth_week'].astype(int)
    return df

def crosstab(df, col1, col2, col3=None, aggfunc=np.mean, **kwargs):
    df = df.copy()

    if col3 is None:
        return pd.crosstab(df[col1], df[col2], **kwargs)
    else:
        return pd.crosstab(df[col1], df[col2], df[col3], aggfunc=aggfunc, **kwargs)

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

def merge_all(df, df_list, on='user_id'):
    for i in df_list:
        df = df.merge(i, on=on, how='left')
    return df

def missing_indicator(df, col):
    df = df.copy()
    df.loc[:, '%s_missing' % col] = df[col].apply(lambda x: 1 if pd.isnull(x) else 0)
    return df

def load(filename, date_cols, folder='/Users/alexhuang/Documents/data/gobble_data/'):
    return pd.read_csv(folder + filename, parse_dates=date_cols)

######
def interactions(df, cols=None):
    df = df.copy()

    if cols:
        df = df[cols]

    for i, j in list(itertools.combinations(df.columns, 2)):
        df['%s*%s' % (i, j)] = df[i] * df[j]

    return df
# add generic transformation function?
def log_transform(df, cols):
    df = df.copy()
    df[cols] = df[cols].apply(lambda x: np.log(x + 1))
    return df
def bin_transform(df, cols):
    df = df.copy()
    df[cols] = df[cols].apply(lambda x: pd.cut(x, 4).cat.codes)
    return df
def add_column(df, col, name=None):
    col = pd.DataFrame(col)
    if name:
        col.columns = [name]
    return pd.concat([df.reset_index(drop=True), col], axis=1)

def stack_all(df_list):
    for i, df in enumerate(df_list):
        df['group'] = i
    return pd.concat(df_list)

# needs user_id and date
def time_diff(df):
    df = df.copy()
    df = df.sort_values(by=['user_id', 'date'])
    df['time_diff'] = df['date'].diff().dt.total_seconds()
    df.loc[df['user_id'] != df['user_id'].shift(1), 'time_diff'] = np.nan
    return df

# have defaults: group='user_id', col='id'?
def total_count(df, group='user_id', col='id'):
    return df.groupby(group)[col].count().reset_index()

def total_value(df, group='user_id', col='id'):
    return df.groupby(group)[col].sum().reset_index()

# need id column?
def average_value(df, col='id'):
    df = df.copy()
    df = df.groupby('user_id')[col].mean().reset_index()
    return df

# needs date filter_start, filter_end, start, end column
def frequency(df, group, col):
    df = df.copy()
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

def col_in(df, col, values):
    return df[df[col].isin(values)]

def col_between(df, left, col, right):
    return df[(df[col] > left) & (df[col] < right)]

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
    df[name] = df[date_col].dt.to_period('W').dt.start_time + DateOffset(**kwargs)
    return df

def add_date_offset(df, date_col, name='next', **kwargs):
    df = df.copy()
    df[name] = df[date_col].dt.date + DateOffset(**kwargs)
    return df

def count_rows(df, group='user_id'):
    a = df.groupby(group).size().reset_index()
    a.columns = [group, 'events']
    return a

def transform_column(df, cols, f):
    df = df.copy()
    df[cols] = df[cols].apply(f)
    return df

def get_dups(df, col):
    a = df.groupby(col).size()
    dup = a[a > 1].index
    return df[df[col].isin(dup)].sort_values(by=col)

def add_agg_col(df, group, col, func):
    df = df.copy()
    df[func] = df.groupby(group)[col].transform(func)
    return df.groupby(group).head(1)

def add_agg_count_col(df, group, func):
    df = df.copy()
    df['count'] = df.groupby(group).transform('count').iloc[:, -1]
    return df.groupby(group).head(1)

def consecutive_runs(df):
    df = df.copy()
    df = df.pipe(mark_nth_week)
    df = df[~df['nth_week'].isnull()]
    df = df.pipe(mark_consecutive_runs, 'nth_week')
    df = df.groupby(['user_id', 'run']).size().reset_index().drop('run', axis=1)
    return df

# need error checking on preserve
def my_query(df, query, on='user_id', preserve='group'):
    return df[[on, preserve]].merge(df.query(query).pipe(remove, [preserve]), on=on, how='left')

def contains_any(a, strings):
    return any([x for x in strings if x in i])

def contains_none_of(a, strings):
    return bool(1 - contains_any(a, strings))

# remove include_target
def get_columns_with(df, include=None, exclude=None, include_target=True):
    df = df.copy()

    if include:
        include = set(include)
    else:
        include = set(df.columns)

    if exclude:
        exclude = set(exclude)
    else:
        exclude = set()

    if include_target:
        include.add('target')

    c = [i for i in df.columns if contains_any(i, include) and contains_none_of(i, exclude)]

    return df[c]

def name(df, names):
    df = df.copy()
    if isinstance(names, (list, tuple)):
        df.columns = ['user_id'] + list(names)
    else:
        df.columns = ['user_id', names]
    return df

def name_with_template(df, template):
    df = df.copy()
    col_names = df.columns.difference(['user_id'])
    return df.pipe(name, ['%s_%s' % (template, i) for i in col_names])

# have functions only return relevant columns
def count_categorical(df, col):
    df = df.copy()
    df = df.pipe(dummies, col).groupby('user_id').sum().reset_index()
    return df

def convert_to_date(df, column):
    df = df.copy()
    df[column] = pd.to_datetime(df[column].dt.strftime('%Y-%m-%d'))
    return df

def fill_in_time_diff(df):
    df = df.copy()
    a = (datetime.now() - df['date']).dt.days
    df['time_diff'] = df['time_diff'].fillna(a)
    return df

def name_append(df, x, ignore=['user_id']):
    df = df.copy()
    col = df.columns.difference(ignore)
    return df.rename(columns=dict(zip(col, [i + '_%s' % x for i in col])))
