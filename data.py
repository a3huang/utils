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
