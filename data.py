from collections import defaultdict
from pandas.tseries.offsets import DateOffset
from datetime import datetime
from scipy.sparse import coo_matrix
from sklearn.datasets import load_iris
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, RFECV, SelectKBest
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sqlalchemy import create_engine
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import boto3
import collections
import importlib
import itertools
import os
import random


def dummy_categorical(df, n, shuffle=False):
    '''
    Creates a categorical column with labels 1 to n for testing
    purposes.

    ex) df['dummy'] = df.pipe(dummy_categorical, 5)
    '''

    size = df.shape[0] / n
    remainder = df.shape[0] - n * size

    lst = []
    lst.append(np.full((size + remainder, 1), 1))

    for i in range(2, n+1):
        lst.append(np.full((size, 1), i))

    a = pd.DataFrame(np.append(lst[0], lst[1:]))

    if shuffle:
        a = np.random.permutation(a)

    return a


def dummy_continuous(df, loc=0, scale=1):
    '''
    Creates a continuous column with each value drawn from a normal
    distribution for testing purposes.

    ex) df['dummy'] = df.pipe(dummy_continuous)
    '''

    values = np.random.normal(loc=loc, scale=scale, size=df.shape[0])
    return pd.DataFrame(values)


def round_to_nearest_mult(x, mult):
    '''
    Rounds a floating point number to the nearest multiple of mult.

    Note: Uses "round half to even" method.

    ex) round_to_nearest_mult(0.0250, 0.05) -> 0
    ex) round_to_nearest_mult(0.0251, 0.05) -> 0.05
    ex) round_to_nearest_mult(1250, 500) -> 1000
    ex) round_to_nearest_mult(1251, 500) -> 1500
    '''

    return mult * np.round(float(x) / mult)


def disjoint_intervals(start, end, step=2):
    '''
    Create 2-tuples of integers where each end point is equal to the next
    start point. Can be used to represent disjoint time intervals.

    ex) disjoint_intervals(0, 6, 2) -> [(0, 2), (2, 4), (4, 6)]
    ex) disjoint_intervals(0, 6, 3) -> [(0, 3), (3, 6)]
    '''

    left_values = range(start, end+step, step)
    right_values = range(start+step, end+step, step)
    return zip(left_values, right_values)


def undummy(df):
    '''
    Turn a set of dummy indicator variables back into a single
    categorical variable.

    ex) df[['earth', 'air', 'fire', 'water']].pipe(undummy)
    '''

    return df.apply(lambda x: x.idxmax(), axis=1)


def top_levels(s, n=None):
    '''
    Keep the top n most common levels of a categorical variable and
    label the rest as 'other'.

    ex) df['Type'].pipe(top_levels, 5)
    '''

    if n:
        counts = s.value_counts()
        top = counts.iloc[:n].index
        return s.apply(lambda x: x if x in top else 'other')
    else:
        return s


def concat_all(*args):
    '''
    Concatenate a list of columns or dataframes without worrying about
    indices.

    ex) concat_all([X, y, model.predict(X)])
    '''

    dfs = args
    reindexed_dfs = [pd.DataFrame(df).reset_index(drop=True) for df in dfs]
    new_df = pd.concat(reindexed_dfs, axis=1)
    return new_df


def merge_all(dfs, on, how='left'):
    '''
    Merge a list of dataframes together.

    ex) merge_all([df1, df2, df3], on='user_id')
    '''

    df_base = dfs[0]
    for df in dfs[1:]:
        df_base = df_base.merge(df, on=on, how=how)

    return df_base


def filter_users(df, f, user_id_column):
    '''
    Filter rows of a dataframe satisfying a complex boolean expression
    and include rows belonging to the same user even if they do not
    satify the condition.

    ex) df.pipe(filter_users, lambda x: x['source'] == 'google', 'user_id')
    '''

    users = df.loc[f(df), user_id_column].unique()
    return df[df[user_id_column].isin(users)]


def filter_top(df, col, n=5):
    top_values = df[col].value_counts()[:n].index
    return df[df[col].isin(top_values)]


def is_unique(df, col):
    '''
    Check if given column values are unique.

    ex) df.pipe(is_unique, 'user_id')
    '''

    if len(df.groupby(col).size().value_counts()) > 1:
        return False
    else:
        return True


def show_missing(df, normalize=False):
    '''
    Shows the number of missing values for each variable.

    ex) df.pipe(show_missing).iloc[:5].sort_values()
    '''

    a = df.isnull().sum()

    if normalize:
        a = a / df.shape[0]

    return a


def show_constant_variance(df):
    '''
    Shows variables in dataframe that are constant.

    ex) df.pipe(show_constant_var)
    '''

    return df.nunique().pipe(lambda x: x[x == 1])


def time_diff(df, date, user_id):
    '''
    Calculates the time difference between consecutive rows of a date
    variable to obtain the duration of each event in seconds. Marks
    the time differences that occur between users as NaN.

    Note that the duration of the final event of each user cannot be
    measured.

    ex) df.pipe(time_diff, date='date', user_id='customer_id')
    '''

    df = df.copy()
    df = df.sort_values(by=[user_id, date])
    df['time_diff'] = df[date].diff().dt.total_seconds()
    df.loc[df[user_id] != df[user_id].shift(), 'time_diff'] = None
    df['duration'] = df['time_diff'].shift(-1)
    return df.drop('time_diff', 1)


def feature_scores(model, X, attr, sort_abs=False, top=None, label=0):
    '''
    Calculate importance scores for each feature for a given model.

    ex) feature_scores(rf, xtrain, feature_importances_)
    '''

    if 'pipeline' in str(model.__class__):
        model = model.steps[-1][1]

    scores = getattr(model, attr)
    if len(scores.shape) > 1:
        scores = scores[label]
    df = pd.DataFrame(zip(X.columns, scores))

    if sort_abs:
        df['abs'] = np.abs(df[1])
        df = df.sort_values(by='abs', ascending=False)
    else:
        df = df.sort_values(by=1, ascending=False)

    if top:
        return df[:top]
    else:
        return df


def mi_ranking(X, y):
    '''
    Ranks the variables in the dataset according to their mutual
    information scores.

    ex) mi_ranking(xtrain, ytrain)
    '''

    model = SelectKBest(score_func=mutual_info_classif)
    model.fit(X, y)
    return feature_scores(model, X, 'scores_')


def rfe_ranking(model, X, y, random_state):
    '''
    Ranks the variables in the dataset according to when they were
    eliminated via RFE with 5-fold CV AUC as the performance measure.
    A rank of 1 means that the variable should be kept.

    ex) rfe_ranking(model, xtrain, ytrain)
    '''

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    feature_selector = RFECV(model, cv=cv, scoring='roc_auc')
    feature_selector.fit(X, y)

    mean_cv = cross_val_score(model, X, y, cv=cv, scoring='roc_auc').mean()
    print 'Original AUC: %s' % mean_cv
    print 'Best AUC: %s' % feature_selector.grid_scores_.max()

    return feature_scores(feature_selector, X, 'ranking_').sort_values(by=1)


def top_correlations(df, n=None):
    '''
    Shows the top correlated pairs of variables sorted by magnitude.

    ex) df.pipe(top_corr, n=5)
    '''

    df = df.corr()
    upper_triangle = np.triu(np.ones(df.shape).astype(np.bool))
    df = df.where(upper_triangle).stack().reset_index()
    df.columns = ['var1', 'var2', 'corr']
    df['abs'] = df['corr'].abs()

    a = df.pipe(filter, lambda x: abs(x['corr']) != 1)
    a = a.sort_values(by='abs', ascending=False)

    if n:
        return a[:n]
    else:
        return a


def scoring_table(true_vals, preds):
    '''
    Takes a dataframe containing predicted scores for a binary
    classification problem in the 1st column and true labels in the
    2nd column. Creates an aggregated scoring table based on deciles.

    ex) scoring_table(model.predict_proba(xtest)[:, 1], y_test)
    '''

    scores = concat_all(preds, true_vals)
    scores.columns = ['scores', 'target']
    scores = scores.sort_values(by='scores', ascending=False)\
                   .reset_index(drop=True)
    scores['Decile'] = pd.qcut(scores.index, 10, labels=False) + 1

    df = scores.groupby('Decile')['scores'].agg([min, max])
    df['count'] = scores.groupby('Decile').size()
    df['composition'] = df['count'] / float(len(scores))
    df['cumulative'] = df['composition'].cumsum()

    df['count_0'] = scores[scores['target'] == 0].groupby('Decile').size()
    total_non_targets = float(len(scores[scores['target'] == 0]))
    df['composition_0'] = df['count_0'] / total_non_targets
    df['cumulative_0'] = df['composition_0'].cumsum()

    df['count_1'] = scores[scores['target'] == 1].groupby('Decile').size()
    total_targets = float(len(scores[scores['target'] == 1]))
    df['composition_1'] = df['count_1'] / total_targets
    df['cumulative_1'] = df['composition_1'].cumsum()

    df['KS'] = df['cumulative_1'] - df['cumulative_0']
    df['rate'] = df['count_1'] / df['count']
    df['index'] = df['rate'] / (total_targets / float(len(scores))) * 100
    df = df.round(2)

    top_columns = ['scores']*2 + ['Population Metrics']*3 + \
                  ['Non-Target Metrics']*3 + ['Target Metrics']*3 + \
                  ['Validation Metrics']*3

    bottom_columns = ['Min Score', 'Max Score', 'Count', 'Composition',
                      'Cumulative', 'Count', 'Composition',
                      'Cumulative', 'Count', 'Composition',
                      'Cumulative', 'K-S', 'Cancel Rate',
                      'Cancel Index']

    df.columns = pd.MultiIndex.from_tuples(zip(top_columns, bottom_columns))

    return df


def create_engine_from_config(config, section, prefix=None):
    '''
    Takes a config object returned by ConfigParser and returns an
    sqlalchemy engine object for the given database specified in
    the section parameter.
    '''

    host = config.get(section, 'host')
    port = config.get(section, 'port')
    user = config.get(section, 'user')
    password = config.get(section, 'password')
    db = config.get(section, 'db')

    if prefix is None:
        prefix = section

    connection_params = (prefix, user, password, host, port, db)
    connection_string = '%s://%s:%s@%s:%s/%s' % connection_params
    return create_engine(connection_string)


def load_iris_example():
    data, target = load_iris(True)
    df = pd.concat([pd.DataFrame(data), pd.DataFrame(target)], axis=1)
    df.columns = ['sepal_l', 'sepal_w', 'petal_l', 'petal_w', 'type']
    return df


def jaccard_similarity(s1, s2):
    num = len(s1.intersection(s2))
    den = len(s1.union(s2))
    return num/float(den)


def jaccard_similarity_correlation(df, col1, col2):
    col1_vals = df[col1].dropna().unique()
    col2_vals = df[col2].dropna().unique()

    m = col1_vals.shape[0]
    n = col2_vals.shape[0]

    a = np.zeros((m, n))

    for i, j in itertools.product(range(m), range(n)):
        top = (df[col1] == col1_vals[i]) & (df[col2] == col2_vals[j])
        bottom = (df[col1] == col1_vals[i]) | (df[col2] == col2_vals[j])

        numerator = df[top].shape[0]
        denominator = df[bottom].shape[0]

        a[i, j] = numerator / float(denominator)

    a = pd.DataFrame(a, index=col1_vals, columns=col2_vals)
    return a


def mark_nth_week(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['start'] = pd.to_datetime(df['start'])
    df['week'] = (df['date'] - df['start']).dt.days / 7 + 1
    df['week'] = df['week'].astype(int)
    df.loc[df['week'] < 0, 'week'] = 0
    return df


def mark_nth_day(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['start'] = pd.to_datetime(df['start'])
    df['day'] = (df['date'] - df['start']).dt.days
    df['day'] = df['day'].astype(int)
    df.loc[df['day'] < 0, 'day'] = 0
    return df


def inv_dict(d):
    return {v: k for k, v in d.items()}


def dict_multi_key(d, keys):
    return [d[i] for i in keys]


def show_unconvertable_integers(x):
    def try_convert_int(x):
        try:
            return int(x)
        except ValueError:
            return 'Error'

    a = pd.concat([x, x.apply(try_convert_int)], axis=1)
    return a[a.iloc[:, 1] == 'Error'].iloc[:, 0]


def show_unique_events(df, date, user_id, event):
    df = df.copy()
    df = df.sort_values([user_id, date])

    different_event = (df[event] != df[event].shift())
    different_user = df[user_id] != df[user_id].shift()
    df['event_change'] = different_event | different_user
    df['event_id'] = df.groupby(user_id)['event_change'].cumsum()
    return df.groupby([user_id, 'event_id']).head(1)\
             .drop(['event_change', 'event_id'], 1)


def insert_between_str_list(l, seps):
    return ''.join([a+b for a, b in zip(l[:-1], seps)]) + l[-1]


def check_joins(query, engine):
    split_query = query.split('join')
    for i in range(len(split_query)):
        joins = ['join']*i + ['left join']*(len(split_query) - i)
        query = insert_between_str_list(split_query, joins)
        print pd.read_sql_query(query, engine).shape
####


def get_ts_counts(df, start, end, name):
    a = mark_nth_day(df).groupby(['user_id', 'day']).size().unstack()
    missing_days = np.setdiff1d(np.array(range(end)), a.columns)
    a = a.reindex(columns=np.append(a.columns.values, missing_days))\
        .sort_index(1)
    a = a.iloc[:, start:end]
    a.columns = ["day_%s_%s" % (i, name) for i in a.columns]
    return a.reset_index()


def get_ts_sum(df, start, end, col, name):
    a = mark_nth_day(df).groupby(['user_id', 'day'])[col].sum().unstack()
    missing_days = np.setdiff1d(np.array(range(end)), a.columns)
    a = a.reindex(columns=np.append(a.columns.values, missing_days))\
        .sort_index(1)
    a = a.iloc[:, start:end]
    a.columns = ["day_%s_%s" % (i, name) for i in a.columns]
    return a.reset_index()


def timeseries(df, datecol, user_col, freq, aggfunc):
    # if want to start on monday -> choose W-SUN
    a = df.set_index(datecol).to_period(freq).reset_index()\
            .groupby([user_col, datecol]).agg(aggfunc).unstack()
    min_date = df[datecol].min()
    max_date = df[datecol].max()
    date_range = pd.date_range(start=min_date, end=max_date,
                               freq=freq) - DateOffset(days=6)

    date_range = date_range.to_series().astype(str).str\
        .split(' ', expand=True)[0]
    a.columns = a.columns.to_series().astype(str).str\
        .split('/', expand=True)[0]

    missing_days = np.setdiff1d(date_range.values, a.columns.values)
    a = a.reindex(columns=np.append(a.columns.values, missing_days))\
        .sort_index(1)
    return a.reset_index()


def cohort_monthly_retention_table(users, by=None):
    users = users.copy()

    if by:
        cols = ['user_id', 'start', 'end', by]
    else:
        cols = ['user_id', 'start', 'end']

    users = users[cols]

    date_range = (datetime.now() - users['start'].min()).days / 30
    date_buckets = pd.date_range(start=users['start'].min(),
                                 periods=date_range, freq='M')
    date_buckets_names = ['month %s' % i for i in range(1, date_range+1)]
    users = pd.concat([users, pd.DataFrame(columns=date_buckets_names)])
    users.loc[:, date_buckets_names] = date_buckets

    df = users.melt(id_vars=cols, value_vars=users.columns.difference(cols))
    df = df.rename(columns={'value': 'date'})
    still_active = (df['date'].between(df['start'], df['end'])) | \
        (df['end'].isnull())

    df = df[still_active & df['date'] <= datetime.now()]

    start = pd.Grouper(key='start', freq='M')
    date = pd.Grouper(key='date', freq='M')

    if by:
        group = by
    else:
        group = start

    table = df.groupby([group, date]).size().unstack()
    table['new'] = users.groupby(group).size()
    table = table[['new'] + table.columns[:-1].tolist()]
    return table


def fetch_table(name):
    '''
    Decorator that sets the table name for each function used to generate a
    feature. This label will be used by the create_table_feature_dict function
    to determine which event table to use when constructing features.
    '''

    def wrapper(f):
        f.table_name = name
        return f

    return wrapper


def create_table_feature_dict(features_file, folder_name):
    '''
    Function that iterates through a given module that contains function
    definitions for generating features. Creates a dictionary with keys being
    the name of the event table to be used and values being a list of function
    objects to be executed. The resulting dictionary will be passed in to the
    create_dataframe function in the model training file.
    '''

    a = importlib.import_module(features_file, folder_name)
    d = collections.defaultdict(list)

    for i in dir(a):
        item = getattr(a, i)
        if callable(item):
            try:
                d[item.table_name].append(item)
            except KeyError:
                continue

    return d


def plot_feat_error(model, df):
    df = df.copy()
    df.loc[(df.pred == 1) & (df.Survived == 0), 'error'] = 'FP'
    df.loc[df.pred == df.Survived, 'error'] = 'C'
    df.loc[(df.pred == 0) & (df.Survived == 1), 'error'] = 'FN'

    df1 = df.drop('error', 1)
    s = StandardScaler()
    df1 = pd.DataFrame(s.fit_transform(df1), columns=df1.columns)
    df2 = concat_all(df1, df['error'])
    sns.heatmap(df2.groupby('error').mean())


def ts_train_test_split(x, val_size=0.2, test_size=0.2):
    val_size = int(len(x) * val_size)
    test_size = int(len(x) * test_size)
    train_size = len(x) - val_size - test_size
    train, val, test = x[0:train_size], \
        x[train_size:(train_size + val_size)],\
        x[(train_size + val_size):]
    return train, val, test


def ts_create_target(df, lag=1):
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.columns = range(len(df.columns))
    df = df.fillna(0)
    return df


def reshape_for_rnn(x, y):
    x = np.array(x)
    y = np.array(y)

    if len(x.shape) == 1:
        x_reshaped = x.reshape(x.shape[0], 1, 1)
    else:
        x_reshaped = x.reshape(x.shape[0], 1, x.shape[1])

    y_reshaped = y

    return x_reshaped, y_reshaped


def undiff(s, c):
    s = pd.Series(s)
    s[0] = c
    return s.cumsum()


def undiff2(s, history):
    s = pd.Series(s)
    last_obs = len(history) - len(s)
    s[0] = history[last_obs]
    return s.cumsum()


def get_conversion_dict(a):
    items = np.sort(a.unique())
    item2id = dict(zip(items, range(len(items))))
    return a.map(item2id).values, item2id


def sparse_crosstab(df, col1, col2, col3):
    col1_ids, col1_dict = get_conversion_dict(df[col1])
    col2_ids, col2_dict = get_conversion_dict(df[col2])
    return coo_matrix((df[col3].values, (col1_ids, col2_ids)),
                      shape=(len(col1_dict), len(col2_dict)))


def input_columns(*columns_lists):
    def decorator(f):
        def wrapper(*args):
            for columns, df in zip(columns_lists, args):
                if columns is None:
                    continue
                if not set(columns).issubset(df.columns):
                    missing_columns = list(set(columns) - set(df.columns))
                    message = "Input missing columns: %s" % missing_columns
                    raise Exception(message)
            return f(*args)
        return wrapper
    return decorator


def output_columns(columns):
    def decorator(f):
        def wrapper(*args):
            output = f(*args)
            if not set(columns).issubset(output.columns):
                missing_columns = list(set(columns) - set(output.columns))
                message = "Output missing columns: %s" % missing_columns
                raise Exception(message)
            return output
        return wrapper
    return decorator


def cancel_after_event_table(w, c):
    w['date_str'] = w.date.dt.to_period('W').map(str).apply(
        lambda x: x.split('/')[0])
    c['date_str'] = c.date.dt.to_period('W').map(str).apply(
        lambda x: x.split('/')[0])

    weeks = c['date_str'].unique()

    lst = []
    for week in weeks:
        users = c[c.date_str == week]['user_id'].unique()
        future_weeks = [pd.to_datetime(week) + DateOffset(weeks=i)
                        for i in range(1, 5)]

        row = []
        for i in future_weeks:
            skip_count = w[(w.user_id.isin(users)) & (w.date == i) &
                           (w.cancelled == 1)].shape[0]
            skip_rate = skip_count / float(len(users))
            row.append(skip_rate)

        lst.append(row)

    df = pd.DataFrame(lst, index=weeks,
                      columns=['%s_week_later' % i for i in range(1, 5)])
    return df


def select_from_iterable(iterable, indices):
    count = 0
    elements = []

    for x in iterable:
        if len(indices) == 0:
            break
        if count == indices[0]:
            elements.append(x)
            indices.pop(0)
        count += 1

    return elements


def sample(it, length, k):
    indices = random.sample(xrange(length), k)
    result = [None]*k
    for index, datum in enumerate(it):
        if index in indices:
            result[indices.index(index)] = datum
    return result


def reservoir_sample(iterable, n):
    results = []
    iterator = iter(iterable)

    # take the first n elements of the iterator
    try:
        for i in xrange(n):
            results.append(iterator.next())
    except StopIteration:
        raise ValueError("Sample larger than population.")

    random.shuffle(results)

    # probability of keeping new item decreases with time
    for i, v in enumerate(iterator, n):
        r = random.randint(0, i)
        if r < n:
            results[r] = v

    return results


def itershuffle(iterable, bufsize=1000):
    iterable = iter(iterable)
    buf = []
    try:
        while True:
            # add elements to a buffer
            for i in xrange(random.randint(1, bufsize-len(buf))):
                buf.append(iterable.next())

            # shuffle the buffer
            random.shuffle(buf)

            # yield elements from the buffer until empty
            for i in xrange(random.randint(1, bufsize)):
                if buf:
                    yield buf.pop()
                else:
                    break

            # go back to iterator and add more elements to buffer
            # break out of while loop when we run out of elements
            # from original iterator
    except StopIteration:
        random.shuffle(buf)

        while buf:
            yield buf.pop()

        raise StopIteration


def remove_all_except(directory, keep):
    to_remove = [i for i in os.listdir(directory) if i not in keep]
    for i in to_remove:
        os.remove(os.path.join(directory, i))


def copy_multiple_files(from_bucket, to_bucket, to_folder, subfolder,
                        parent_folder):
    result = boto3.resource('s3').meta.client.list_objects_v2(
        Bucket=from_bucket, MaxKeys=1000,
        Prefix=os.path.join(parent_folder, subfolder))

    for content in result['Contents']:
        parent_levels = len(parent_folder.split('/'))
        folderfile = os.path.join(*content['Key']
                                  .split('/')[-parent_levels:])

        boto3.resource('s3').meta.client.copy(CopySource={
                    'Bucket': from_bucket,
                    'Key': content['Key'],
                }, Bucket=to_bucket, Key=os.path.join(to_folder, folderfile))


def rename_history(root_folder, subfolder, new_prefix):
    filename = [i for i in os.listdir(os.path.join(root_folder, subfolder))
                if i.endswith('_history.csv')][0].split('.')[0]

    original = os.path.join(root_folder, subfolder, filename + '.csv')
    new = os.path.join(root_folder, subfolder, filename + '_' +
                       new_prefix + '.csv')
    os.rename(original, new)


def upload_files(bucket, root_folder, subfolder):
    for i in os.listdir(os.path.join(root_folder, subfolder)):
        local_path = os.path.join(root_folder, subfolder, i)
        file_part = os.path.join(*local_path.split('/')[-2:])
        destination = os.path.join('fenix/data', file_part)

        boto3.resource('s3').meta.client.upload_file(local_path, bucket,
                                                     destination)


def read_multi_csv(folder):
    paths = os.listdir(folder)
    lst = []
    for i in paths:
        lst.append(pd.read_csv(os.path.join(folder, i), sep='|'))
    df = pd.concat(lst)
    return df


def abs_listdir(root_dir, s):
    return [os.path.join(root_dir, s, i)
            for i in os.listdir(os.path.join(root_dir, s))]


def get_feat_importance(model):
    attributes = ['coef_', 'feature_importances_']

    if hasattr(model, 'steps'):
        model = model.steps[-1][-1]

    for i in attributes:
        if hasattr(model, i):
            feat_attr = i
            break

    return getattr(model, feat_attr)


def plot_multistep_forecast(s, preds, k=100, h=4, step=4):
    plt.plot(s.values, color='b')
    for i in range(1, len(s)-k-h, step=4):
        blanks = np.array([None]*(k+i-2))
        prev_true = s.values[k+i-2:k+i]
        current_preds = np.array(preds)[i-1]
        plt.plot(np.concatenate([blanks, prev_true, current_preds]), color='g')


def get_forecast_errors(s, predict_func, k=100, h=4):
    l1 = []
    l2 = []
    for i in range(1, len(s)-k-h+1):
        train_on = s[:(k+i-1)]
        test_on = s[(k+i-1):(k+i-1)+h].values

        pred = predict_func(train_on, h)

        l1.append(np.sum((test_on - pred)**2))
        l2.append(pred)

    return np.array(l1).mean(), np.array(l2)


def get_forecast_errors2(s, predict_func, train_end=100, h=4):
    l1 = []
    l2 = []
    for i in range(1, len(s)-train_end-h+1):
        train_length = train_end + i - 1
        train_on = s[:train_length]
        test_on = s[train_length:train_length + h].values

        pred = predict_func(train_on, h)

        l1.append(np.sum((test_on - pred)**2))
        l2.append(pred)

    return np.array(l1).mean()


def get_regression_forecast_errors(X, y, model, k=100, h=4):
    l1 = []
    l2 = []
    for i in range(1, len(X)-k-h+1):
        X_train_on = X[:(k+i-1)]
        y_train_on = y[:(k+i-1)]

        X_test_on = X[(k+i-1):(k+i-1)+h]
        y_test_on = y[(k+i-1):(k+i-1)+h]

        model.fit(X_train_on, y_train_on)
        pred = model.predict(X_test_on)

        l1.append(np.sum((y_test_on - pred)**2))
        l2.append(pred)

    return np.array(l1).mean(), np.array(l2)


def naive_predict(train_on, h, *args):
    return train_on.iloc[-h:]


def average_predict(train_on, h, *args):
    return [train_on.mean()]*h


def moving_average_predict(train_on, h, *args):
    return train_on.rolling(14).mean().iloc[-h:].values


def exp_smooth_predict(train_on, h, *args):
    ses = SimpleExpSmoothing(train_on.values).fit(
        smoothing_level=.1, optimized=False)
    return ses.forecast(h)[-h:]


def holt_predict(train_on, h, *args):
    holt_model = Holt(train_on.values).fit(
        smoothing_level=0.1, smoothing_slope=0.01)
    return holt_model.forecast(h)[-h:]


def holt_winters_predict(train_on, h, *args):
    es_model = ExponentialSmoothing(
        train_on.values, seasonal_periods=12, trend='additive',
        seasonal='multiplicative').fit()
    return es_model.forecast(h)[-h:]


def fourier_series(dates, p, n):
    # each column is the series expansion at s(t)
    seconds_since_epoch = np.array((dates - pd.datetime(1970, 1, 1))
                                   .dt.total_seconds().astype(np.float))
    t = seconds_since_epoch / (3600 * 24.)

    fourier = [f((2.0 * (i + 1) * np.pi * t / p))
               for i in range(n) for f in (np.sin, np.cos)]

    return np.concatenate(fourier)


def piecewise_linear_cumulative(x, a, c):
    # evaluates function of form:
    # sum_(k) [a_k * (x - c_k) * 1_(x > c_k)]

    # thus the result will look like:
    # 0
    # 0
    # ...
    # a_1 * (x_5 - c_1)
    # a_1 * (x_6 - c_1)
    # ...
    # a_1 * (x_10 - c_1) + a_2 * (x_10 - c_2)

    break_points = np.concatenate(([0], c, [len(x)]))
    diffs = np.ediff1d(break_points)

    g = a * c

    a_cum = np.cumsum(a)
    g_cum = np.cumsum(g)

    a_k = np.repeat(np.concatenate(([0], a_cum)), diffs)
    g_k = np.repeat(np.concatenate(([0], g_cum)), diffs)

    # a_k * x - a_k * c_k = a_k * (x - c_k)
    return a_k * x - g_k


def delta_functions_cdf(w, x):
    on_weights = np.array([i for i in w if i < x])
    return np.sum(on_weights)


def inverse_sample(x, cdf):
    mask = (np.random.uniform() < cdf)
    min_sub_idx = np.argmin(cdf[mask])
    min_idx = np.arange(cdf.shape[0])[mask][min_sub_idx]
    return x[min_idx]


def lag_feature(df, group, col, lags):
    df = df.copy()
    for i in range(1, lags+1):
        df.loc[:, 'lagged_%s_%s' % (col, i)] = df.groupby(group)[col].shift(i)
    return df


def error_per_category(categories, test_vectors, pred_vectors):
    # test_vectors and pred_vectors are list of series
    lst = []
    for c, i in zip(categories, range(len(test_vectors))):
        a, b = test_vectors[i].align(pred_vectors[i], join='inner')
        print "%s %s" % (c, np.sqrt(np.mean((a - b)**2)))
        lst.extend(a - b)
    lst = np.array(lst)

    return np.sqrt(np.nanmean(lst**2))


def plot_ts_by_category(categories, test_vectors, pred_vectors):
    fig, ax = plt.subplots(len(categories)/3, 3, figsize=(14, 10))

    for ax_i, i, c in zip(ax.flatten(), range(len(test_vectors)),
                          categories):
        ax_i.plot(test_vectors[i], label='True')
        ax_i.plot(pred_vectors[i], label='Estimate')
        ax_i.set_title(c)
        plt.setp(ax_i.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    ax_i.legend()

    # delete any unused axes
    for i in range(1, len(ax.flatten()) - len(test_vectors) + 1):
        fig.delaxes(ax.flatten()[-i])


def window_feature(df, group, col, lengths):
    df = df.copy()
    for i in lengths:
        if i == -1:
            feature = df.groupby(group)[col].expanding(window=i).mean().values
            df.loc[:, 'windowed_%s_all' % (col,)] = feature
        else:
            feature = df.groupby(group)[col].rolling(window=i).mean().values
            df.loc[:, 'windowed_%s_%s' % (col, i)] = feature
    return df


def ts_grid_search(X, y, model, params, k):
    d = {}
    best_params = None
    for param in itertools.product(*params.values()):
        param_dict = dict(zip(params.keys(), param))
        cur_params = tuple(param_dict.values())

        for p, v in param_dict.items():
            setattr(model, p, v)

        result, _ = get_regression_forecast_errors(X, y, model, k=k)
        d[cur_params] = np.sqrt(np.mean(result))

        if best_params is None:
            best_params = cur_params
        else:
            if d[cur_params] < d[best_params]:
                best_params = cur_params
    return d, best_params


def get_regression_forecasts(X, y, model, k=100, h=4, skip=4):
    errors = []
    l1 = []
    l2 = []
    for i in range(1, (len(X)-k-h+1)+1, skip):
        X_train_on = X[:(k+i-1)]
        y_train_on = y[:(k+i-1)]

        X_test_on = X[(k+i-1):(k+i-1)+h]
        y_test_on = y[(k+i-1):(k+i-1)+h]

        model.fit(X_train_on.drop('ds', 1), y_train_on)
        pred = model.predict(X_test_on.drop('ds', 1))

        errors.append(np.mean((y_test_on - pred)**2))
        y_test_on = pd.Series(y_test_on)
        y_test_on.index = X_test_on.ds
        l1.append(y_test_on)

        pred = pd.Series(pred)
        pred.index = X_test_on.ds
        l2.append(pred)

    return errors, pd.concat(l1), pd.concat(l2)


def get_prophet_forecast_errors(data, model_builder, k=100, h=4, skip=4):
    errors = []
    l1 = []
    l2 = []
    for i in range(1, (len(data)-k-h+1)+1, skip):
        train_on = data[:(k+i-1)]
        test_on = data[(k+i-1):(k+i-1)+h]

        m = model_builder()
        m.fit(train_on)

        pred = m.predict(test_on).set_index('ds').yhat
        test_on = test_on.set_index('ds').y

        errors.append(np.mean((test_on - pred)**2))
        l1.append(test_on)
        l2.append(pred)

    return errors, pd.concat(l1), pd.concat(l2)


def plot_forecasts(X, y, model, k=100, h=4):
    # requires ds column containing datetime stamp
    test_vals, pred_vals = get_regression_forecasts(X, y, model, k, h)
    fig, ax = plt.subplots()
    test_vals.groupby('ds').head(1).plot(ax=ax, label='True')
    pred_vals.groupby('ds').apply(list).apply(pd.Series)\
        .plot(ax=ax, label='Estimate')
    plt.legend()


def stacked_area_plot_for_n_step_errors(preds, test):
    actuals = test.set_index('ds')['y']
    errors = [abs(preds.iloc[i::4] - actuals) for i in range(4)]
    pd.concat(errors, axis=1).plot(kind='area', stacked=True)


def without_keys(d, keys):
    return {x: d[x] for x in d if x not in keys}


def simple_kalman_filter(s, Q=1e-5, R=.1**2, xhat0=0, P0=1):
    n = len(s)

    xhat = np.zeros(n)
    P = np.zeros(n)
    xhatminus = np.zeros(n)
    Pminus = np.zeros(n)
    K = np.zeros(n)

    xhat[0] = xhat0
    P[0] = P0

    for k in range(1, n):
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + Q

        # measurement update
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (s[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]

    return xhat


def plot_sample_gp(kernel, n):
    gp = GaussianProcessRegressor(kernel=kernel, random_state=12)
    x = np.linspace(-1, 1, 100).reshape(-1, 1)
    y = gp.sample_y(x, n)
    plt.plot(x, y)


def plot_train_test_ts(model, X_train, X_test, y_train, y_test):
    y = pd.concat([y_train, y_test])

    plt.plot(range(len(y)), y, label='true')

    plt.plot(range(len(y_train)), model.predict(X_train),
             label='predict_train')

    plt.plot([len(y_train)-1, len(y_train)],
             model.predict(pd.concat([X_train.iloc[[-1]], X_test.iloc[[0]]])))

    plt.plot(range(len(y_train), len(y)), model.predict(X_test),
             label='predict_test')

    plt.legend()


def cond_drop(df, columns):
    for col in columns:
        if col in columns:
            df = df.drop(col, 1)
    return df


def add_word(d, s):
    for i in range(len(s)-1):
        d[(s[i], s[i+1])] += 1

    return d


def get_proportions(d):
    t = defaultdict(list)
    for i in d:
        t[i[0]].append((i[1], d[i]))

    for i in t:
        t[i] = (sum([j[1] for j in t[i]]), t[i])
        t[i] = (t[i][0], [(j[0], j[1] / t[i][0]) for j in t[i][1]])
    return t


def generate_word(t, n):
    s = np.random.choice(list(t.keys()))
    for i in range(n-1):
        s = s + np.random.choice([i[0] for i in t['n'][1]],
                                 p=[i[1] for i in t['n'][1]])
    return s


def quick_reload(module_name):
    module = importlib.import_module(module_name)
    importlib.reload(module)
