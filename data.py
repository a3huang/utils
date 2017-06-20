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

def crosstab(df, col1, col2, col3=None, **kwargs):
    df = df.copy()

    if col3 is None:
        return pd.crosstab(df[col1], df[col2], **kwargs)
    else:
        return pd.crosstab(df[col1], df[col2], df[col3], aggfunc=np.mean, **kwargs)

def dummies(df, col):
    df = df.copy()
    dummy_col = pd.get_dummies(df[col])
    dummy_col.columns = [str(i) for i in dummy_col.columns]
    df = pd.concat([df.drop(col, 1), dummy_col], axis=1)
    return df

# target.contains_any(strings)
def contains_any(target, strings):
    return any([x for x in strings if x in target])
