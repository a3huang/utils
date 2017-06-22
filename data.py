import itertools
import numpy as np
import pandas as pd

def requires_col(cols):
    def decorator(f):
        def wrapper(df, *args, **kwargs):
            for i in cols:
                if i not in df.columns:
                    raise ValueError('df needs column named %s' % i)
            f(df, *args, **kwargs)
        return wrapper
    return decorator

@requires_col(['date', 'start'])
def filter_first_n_weeks(df, n):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['filter_start'] = df['start']
    df['filter_end'] = df['start'] + DateOffset(weeks=n)
    df = df.query('filter_start <= date < filter_end')
    return df

@requires_col(['date', 'start'])
def filter_last_n_weeks(df, n):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['filter_start'] = datetime.now() - DateOffset(months=n)
    df['filter_end'] = datetime.now()
    df = df.query('filter_start <= date < filter_end')
    return df

@requires_col(['date', 'start'])
def filter_week_window(df, n1, n2):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['filter_start'] = df['start'] + DateOffset(weeks=n1)
    df['filter_end'] = df['start'] + DateOffset(weeks=n2)
    df = df.query('filter_start <= date < filter_end')
    return df

# df should have user_id column
def mark_adjacent_groups(df, col):
    df = df.copy()
    is_diff_number = df[col] != df[col].shift()
    is_diff_user = df['user_id'] != df['user_id'].shift()
    df['group'] = (is_diff_number | is_diff_user).cumsum()
    return df

# df should have user_id column
def mark_consecutive_runs(df, col):
    df = df.copy()
    is_nonconsecutive_number = df[col] != df[col].shift() + 1
    is_diff_user = df['user_id'] != df['user_id'].shift()
    df['run'] = (is_nonconsecutive_number | is_diff_user).cumsum()
    return df

@requires_col(['date', 'start'])
def mark_nth_week(df):
    df = df.copy()
    df['nth_week'] = (df['date'] - df['start']).dt.days / 7 + 1
    df['nth_week'] = df['nth_week'].astype(int)
    return df

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

def add_column(df, col, name):
    col = pd.DataFrame(col)
    col.columns = [name]
    return pd.concat([df.reset_index(drop=True), col], axis=1)

def crosstab(df, col1, col2, col3=None, aggfunc=np.mean, **kwargs):
    df = df.copy()

    if col3 is None:
        return pd.crosstab(df[col1], df[col2], **kwargs)
    else:
        return pd.crosstab(df[col1], df[col2], df[col3], aggfunc=aggfunc, **kwargs)

def dummies(df, col):
    df = df.copy()
    dummy_col = pd.get_dummies(df[col])
    dummy_col.columns = [str(i) for i in dummy_col.columns]
    df = pd.concat([df.drop(col, 1), dummy_col], axis=1)
    return df

# target.contains_any(strings)
def str_contains(target_str, list_of_str):
    return any([x for x in list_of_str if x in target_str])

# df.pipe(merge_all, [df1, df2, df3], **kwargs)
def merge_all(df, df_list):
    for i in df_list:
        df = df.merge(i, **kwargs)
    return df

def stack_all(df_list):
    for i, df in enumerate(df_list):
        df['group'] = i
    return pd.concat(df_list)

def remove(df, cols):
    return df[df.columns.difference(cols)]

# needs user_id and date
def time_diff(df):
    df = df.copy()
    df = df.sort_values(by=['user_id', 'date'])
    df['time_diff'] = df['date'].diff().dt.total_seconds()
    df.loc[df['user_id'] != df['user_id'].shift(1), 'time_diff'] = np.nan
    return df

# have defaults: group='user_id', col='id'?
def agg_total_count(df, group, col):
    return df.groupby(group)[col].count().reset_index()

def agg_total_value(df, group, col):
    return df.groupby(group)[col].sum().reset_index()

# needs date filter_start, filter_end, start, end column
def agg_frequency(df, group, col):
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

# should refactor filtering part
def get_grouped_rates(df, group, col):
    l = []
    group_val = df[group].unique()
    for i in group_val:
        l.append(df[(df[group] == i) & (~df[col].isnull())].shape[0] / float(df[df[group] == i].shape[0]))
    return l

def get_rate(df, col):
    a = df[col].value_counts(dropna=False)
    return a / float(sum(a))

# add default "names" to all other functions as well
def add_dow_offset(df, date_col, name='next_dow', weeks=1, dow=0):
    df = df.copy()
    df[name] = a['date'].dt.to_period('W').dt.start_time + DateOffset(weeks=weeks, days=dow)
    return df

def add_date_offset(df, date_col, name='next', **kwargs):
    df = df.copy()
    df[name] = df[date_col].dt.date + DateOffset(**kwargs)
    return df

def count_rows(df, group='user_id'):
    a = df.groupby(group).size().reset_index()
    a.columns = [group, 'events']
    return a
