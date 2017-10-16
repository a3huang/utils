from datetime import datetime
from pandas.tseries.offsets import *

from boruta import BorutaPy
from sklearn.base import TransformerMixin
from sklearn.feature_selection import mutual_info_classif, RFECV, SelectKBest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine

import numpy as np
import pandas as pd

def disjoint_intervals(start, end, step=2):
    '''
    Create 2-tuples of integers where each end point is equal to the next
    start point. Can be used to represent disjoint time intervals.

    ex) disjoint_intervals(0, 6, 2) -> [(0, 2), (2, 4), (4, 6)]
    ex) disjoint_intervals(0, 6, 3) -> [(0, 3), (3, 6)]
    '''

    return zip(range(start, end+step, step), range(start+step, end+step, step))

def dummy(a):
    '''
    Turn a categorical variable into indicator variables for each level.

    ex) df['Type'].pipe(dummy)
    '''

    df = pd.get_dummies(a)
    df.columns = [str(i) for i in df.columns]
    return df

def undummy(a):
    '''
    Turn a set of dummy indicator variables back into a categorical variable.

    ex) df[['earth', 'air', 'fire', 'water']].pipe(undummy)
    '''

    return a.apply(lambda x: x.idxmax(), axis=1)

def dummy_replace(df, cols):
    '''
    Create dummy indicators for each categorical column specified and replace
    the original columns with dummy variables for each level.

    ex) df.pipe(dummy_replace, cols=['Type 1', 'Type 2'])
    '''

    a = pd.get_dummies(df, columns=cols, dummy_na=True)

    nan_cols = [i for i in a.columns if '_nan' in i]
    for col in nan_cols:
        if len(a[col].value_counts()) == 1:
            a = a.drop(col, 1)

    return a

def time_unit(a, unit):
    '''
    Extract the value of a date variable with respect to a given time unit.

    ex) df['date'].pipe(time_unit, 'weekday')
    '''

    return getattr(a.dt, unit)

def top(a, n=None):
    '''
    Keep the top n most common levels of a categorical variable and label the
    rest as 'other'.

    ex) df['Type'].pipe(top, 5)
    '''

    if n:
        counts = a.value_counts()
        top = counts.iloc[:n].index
        return a.apply(lambda x: x if x in top else 'other')
    else:
        return a

def encode_str(a):
    '''
    Encodes a string variable into an integer variable for use in ML algorithms.

    ex) df['status'].pipe(encode_str)
    '''

    encoder = LabelEncoder()
    return encoder.fit_transform(a)

def cbind(*args):
    '''
    Horizontally concatenate a list of columns or dataframes together without
    worrying about indices.

    ex) cbind(X, y, model.predict(X))
    '''

    df_list = args
    df = pd.concat([pd.DataFrame(df).reset_index(drop=True) for df in df_list], axis=1)

    if len(df.columns.value_counts().pipe(filter, lambda x: x > 1)) > 0:
        print '[WARNING]: May contain duplicate columns.'

    return df

def merge(df_list, on, how, **kwargs):
    '''
    Merge a list of dataframes together.

    ex) merge([df1, df2, df3], on='user_id', how='left')
    '''

    df = df_list[0]
    for df_i in df_list[1:]:
        df = df.merge(df_i, on=on, how=how, **kwargs)
    return df

def filter(df, f):
    '''
    Filter rows of aa dataframe satisfying a complex boolean expression without
    having to specify its name.

    ex) df.pipe(filter, lambda x: x['date'] > '2017-01-01')
    '''

    return df[f(df)]

def filter_i(df, f, name):
    '''
    Filter rows of a dataframe satisfying a complex boolean expression and
    create an indicator variable equal to 1 when the condition is true and 0
    otherwise.

    ex) df.pipe(filter_i, lambda x: x['Attack'] > 200 , 'Strong')
    '''

    df = df.copy()
    df.loc[f(df), name] = 1
    df.loc[:, name] = df.loc[:, name].fillna(0)
    return df

def filter_u(df, f, user_id):
    '''
    Filter rows of a dataframe satisfying a complex boolean expression and
    include rows belonging to the same user even if they do not satify the
    condition.

    ex) df.pipe(filter_u, lambda x: x['source'] == 'google', 'user_id')
    '''

    ids = df.pipe(query, f)[user_id].unique()
    return df[df[user_id].isin(ids)]

def check_unique(df, col):
    '''
    Check if given column values are unique.

    ex) df.pipe(check_unique, 'user_id')
    '''

    if len(df.groupby(col).size().value_counts()) > 1:
        return False
    else:
        return True

def show_duplicates(df, col):
    '''
    Show rows containing duplicate values of a given column.

    ex) df.pipe(show_duplicates, 'user_id')
    '''

    counts = df.groupby(col).size()
    duplicates = counts[counts > 1].index
    return df[df[col].isin(duplicates)].sort_values(by=col)

def show_missing(df, normalize=False):
    '''
    Shows the number of missing values for each variable.

    ex) df.pipe(show_missing).iloc[:5].sort_values().plot.barh()
    '''

    a = df.isnull().sum()

    if normalize == True:
        a = a / df.shape[0]

    return a

def show_constant_var(df):
    '''
    Shows variables in dataframe that are constant.

    ex) df.pipe(show_constant_var)
    '''

    return df.nunique().pipe(filter, lambda x: x == 1)

def show_low_var(df):
    '''
    Shows variables in dataframe sorted increasing from lowest standard
    deviation.

    ex) df.pipe(show_low_var)
    '''

    return df.std().sort_values()

def table(df, row_var, col_var, val_var=None, row_n=None, col_n=None, agg_func=np.mean, **kwargs):
    '''
    Calculate the cross tabulation between 2 categorical variables. Can optionally
    specify how many of the top categories to display in the rows and columns
    of the resulting table.

    ex) df.pipe(table, cat1, cat2, row_n=5, col_n=5)
    '''

    df = df.copy()

    if row_n:
        df[row_var] = df[row_var].pipe(reduce_cardinality, row_n)
    if col_n:
        df[col_var] = df[col_var].pipe(reduce_cardinality, col_n)

    if val_var is None:
        return pd.crosstab(df[row_var], df[col_var], **kwargs)
    else:
        return pd.crosstab(df[row_var], df[col_var], df[val_var], aggfunc=agg_func, **kwargs)

def time_diff(df, date, user_id):
    '''
    Calculates the time difference between consecutive rows of a date variable
    to obtain the duration of each event in seconds. Marks the time differences
    that occur between users as NaN.

    Note that the duration of the final event of each user cannot be measured.

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
    Ranks the variables in the dataset according to their mutual information
    scores.

    ex) mi_ranking(xtrain, ytrain)
    '''

    model = SelectKBest(score_func=mutual_info_classif)
    model.fit(X, y)
    return feature_scores(model, X, 'scores_')

def rfe_ranking(model, X, y, random_state):
    '''
    Ranks the variables in the dataset according to when they were eliminated
    via RFE with 5-fold CV AUC as the performance measure. A rank of 1 means
    that the variable should be kept.

    ex) rfe_ranking(model, xtrain, ytrain)
    '''

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    feature_selector = RFECV(model, cv=cv, scoring='roc_auc')
    feature_selector.fit(X, y)

    print 'Original AUC: %s' % cross_val_score(model, X, y, cv=cv, scoring='roc_auc').mean()
    print 'Best AUC: %s' % feature_selector.grid_scores_.max()

    return feature_scores(feature_selector, X, 'ranking_').sort_values(by=1)

def boruta_ranking(model, X, y, random_state):
    '''
    Ranks the variables in the dataset according to their relevance in
    predicting the target variable. A rank of 1 means that the variable is
    important while a rank of 2 means that it is tenatively important.

    ex) boruta_ranking(model, xtrain, ytrain)
    '''

    model = RandomForestClassifier(n_jobs=-1, max_depth=5, random_state=random_state)
    feature_selector = BorutaPy(model, n_estimators='auto', random_state=random_state)
    feature_selector.fit(X.values, y.values)

    return feature_scores(feature_selector, X, 'ranking_').sort_values(by=1)

def top_corr(df, n=None):
    '''
    Shows the top correlated pairs of variables sorted by magnitude.

    ex) df.pipe(top_corr, n=5)
    '''

    df = df.corr()
    df = df.where(np.triu(np.ones(df.shape).astype(np.bool))).stack().reset_index()
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
    Takes a dataframe containing predicted scores for a binary classification
    problem in the 1st column and true labels in the 2nd column. Creates an
    aggregated scoring table based on deciles.

    ex) scoring_table(cbind(model.predict_proba(xtest)[:, 1], y_test))
    '''

    scores = cbind(preds, true_vals)
    scores.columns = ['scores', 'target']
    scores = scores.sort_values(by='scores', ascending=False).reset_index(drop=True)
    scores['Decile'] = pd.qcut(scores.index, 10, labels=False) + 1

    df = scores.groupby('Decile')['scores'].agg([min, max])
    df['count'] = scores.groupby('Decile').size()
    df['composition'] = df['count'] / float(len(scores))
    df['cumulative'] = df['composition'].cumsum()

    df['count_0'] = scores[scores['target'] == 0].groupby('Decile').size()
    df['composition_0'] = df['count_0'] / float(len(scores[scores['target'] == 0]))
    df['cumulative_0'] = df['composition_0'].cumsum()

    df['count_1'] = scores[scores['target'] == 1].groupby('Decile').size()
    df['composition_1'] = df['count_1'] / float(len(scores[scores['target'] == 1]))
    df['cumulative_1'] = df['composition_1'].cumsum()

    df['KS'] = df['cumulative_1'] - df['cumulative_0']
    df['rate'] = df['count_1'] / df['count']
    df['index'] = df['rate'] / (len(scores[scores['target'] == 1]) / float(len(scores))) * 100
    df = df.round(2)

    top_columns = ['scores']*2 + ['Population Metrics']*3 + ['Non-Target Metrics']*3 + \
                  ['Target Metrics']*3 + ['Validation Metrics']*3
    bottom_columns = ['Min Score', 'Max Score', 'Count', 'Composition', 'Cumulative', 'Count',
                      'Composition', 'Cumulative', 'Count', 'Composition', 'Cumulative', 'K-S',
                      'Cancel Rate', 'Cancel Index']

    df.columns = pd.MultiIndex.from_tuples(zip(top_columns, bottom_columns))

    return df

def compare_data_test(model, datasets, target, omit=None, threshold=0.5, random_state=42):
    '''
    Compares the AUC, confusion matrix, and classification report (precision,
    recall, f1 score) for a given model over several datasets.

    ex) compare_datasets_test(model, [df1, df2, df3, df4, df5], target='cancel',
            omit=['user_id'], threshold=0.1)
    '''

    if omit is None:
        omit = []

    for df in datasets:
        X = df[df.columns.difference(omit + [target])]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
            random_state=random_state)

        model.fit(X_train, y_train)

        pred = model.predict_proba(X_test)[:, 1]
        true = y_test

        print roc_auc_score(true, pred)
        print confusion_matrix(true, pred > threshold)
        print classification_report(true, pred > threshold)
        print

def compare_data_cv(model, datasets, target, omit=None, random_state=42):
    '''
    Compares mean 5-fold CV AUC for a given model over several datasets.

    ex) compare_datasets_cv(model, [df1, df2, df3, df4, df5], target='cancel',
            omit=['user_id'])
    '''

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    if omit is None:
        omit = []

    for df in datasets:
        X = df[df.columns.difference(omit + [target])]
        y = df[target]
        print cross_val_score(model, X, y, cv=cv, scoring='roc_auc').mean()

def compare_model_cv(models, df, target, omit=None, random_state=42):
    '''
    Compares mean 5-fold CV AUC for a given dataset over several models.

    ex) compare_models_cv([model1, model2, model3], df, target='cancel',
            omit=['user_id'])
    '''

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    if omit is None:
        omit = []

    X = df[df.columns.difference(omit + [target])]
    y = df[target]

    for model in models:
        print cross_val_score(model, X, y, cv=cv, scoring='roc_auc').mean()

def compare_model_data_cv(models, datasets, target, omit=None, random_state=42):
    '''
    Compares mean 5-fold CV AUC for all combinations of the given models and
    datasets.

    ex) compare_models_data_cv([model1, model2, model3], [df1, df2, df3],
            target='cancel', omit=['user_id'])
    '''

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    if omit is None:
        omit = []

    scores = []
    for i, df in enumerate(datasets):
        scores_by_df = []
        for model in models:
            X = df[df.columns.difference(omit + [target])]
            y = df[target]
            score = cross_val_score(model, X, y, cv=cv, scoring='roc_auc').mean()
            scores_by_df.append(score)

        scores_df = pd.DataFrame(scores_by_df, columns=['df_%s' % i])
        scores.append(scores_df)

    df = cbind(scores)
    df.index = ['model_%s' % i for i in range(len(models))]
    return df
######

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

# split into cut_width and cut_num?
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

def relative_time_window(df, left_offset, right_offset, frequency):
    '''
    Filter rows of a transactional dataframe with date lying within a relative
    time window.

    ex) df.pipe(relative_time_window, 1, 2, freq='7D')
    '''

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['start'] = pd.to_datetime(df['start'])
    df['left_bound'] = df.set_index('start').shift(periods=left_offset, freq=freq).index
    df['right_bound'] = df.set_index('start').shift(periods=right_offset, freq=freq).index
    df = df.query('left_bound <= date < right_bound')
    return df

def timeseries(df, datecol, user_col, freq, aggfunc):
    # if want to start on monday -> choose W-SUN
    a = df.set_index(datecol).to_period(freq).reset_index()\
            .groupby([user_col, datecol]).agg(aggfunc).unstack()
    min_date = df[datecol].min()
    max_date = df[datecol].max()
    date_range = pd.date_range(start=min_date, end=max_date, freq=freq) - DateOffset(days=6)

    date_range = date_range.to_series().astype(str).str.split(' ', expand=True)[0]
    a.columns = a.columns.to_series().astype(str).str.split('/', expand=True)[0]

    missing_days = np.setdiff1d(date_range.values, a.columns.values)
    a = a.reindex(columns=np.append(a.columns.values, missing_days)).sort_index(1)
    return a.reset_index()

def sample(a, index=False):
    if index == True:
        return a.sample(1).index[0]
    else:
        return a.sample(1).values[0]

def binned_barplot(df, col, bins=5):
    a = pd.cut(df[col], bins=bins).value_counts()
    a.sort_index(ascending=False).plot.barh()

def plot_parallel_coordinates(df, by, cols, n=1000, ax=None):
    df1 = df.sample(n)[cols + [by]]

    s = MinMaxScaler()

    df2 = cbind(pd.DataFrame(s.fit_transform(df1.iloc[:, :-1]), columns=cols), df1[by])

    parallel_coordinates(df2, by, ax=ax)
    plt.xticks(rotation=90)

def plot_radviz(df, by, cols, n=1000, ax=None):
    df1 = df.sample(n)[cols + [by]]
    radviz(df1, by, ax=ax)

def cohort_monthly_retention_table(users, by=None):
    users = users.copy()

    if by:
        cols = ['user_id', 'start', 'end', by]
    else:
        cols = ['user_id', 'start', 'end']

    users = users[cols]

    date_range = (datetime.now() - users['start'].min()).days / 30
    date_buckets = pd.date_range(start=users['start'].min(), periods=date_range, freq='M')
    date_buckets_names = ['month %s' % i for i in range(1, date_range+1)]
    users = pd.concat([users, pd.DataFrame(columns=date_buckets_names)])
    users.loc[:, date_buckets_names] = date_buckets

    df = users.melt(id_vars=cols, value_vars=users.columns.difference(cols))
    df = df.rename(columns={'value': 'date'})
    df = df.pipe(filter, lambda x: (x['date'].between(x['start'], x['end'])) | (x['end'].isnull()))\
           .pipe(filter, lambda x: x['date'] <= datetime.now())

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

def unique_events(df, date, user_id, event):
    df = df.copy()
    df = df.sort_values([user_id, date])
    df['event_change'] = (df[event] != df[event].shift()) | (df[user_id] != df[user_id].shift())
    df['event_id'] = df.groupby(user_id)['event_change'].cumsum()
    return df.groupby([user_id, 'event_id']).head(1).drop(['event_change', 'event_id'], 1)

def jaccard_similarity_table(df, col1, col2):
    col1_vals = df[col1].dropna().unique()
    col2_vals = df[col2].dropna().unique()

    m = col1_vals.shape[0]
    n = col2_vals.shape[0]

    a = np.zeros((m, n))

    for i, j in product(range(m), range(n)):
        top = df[(df[col1] == col1_vals[i]) & (df[col2] == col2_vals[j])].shape[0]
        bot = df[(df[col1] == col1_vals[i]) | (df[col2] == col2_vals[j])].shape[0]
        a[i, j] = top / float(bot)

    a = pd.DataFrame(a, index=col1_vals, columns=col2_vals)
    return a

def model_pred_corrs(models, X):
    scores = []
    for model in models:
        scores.append(model.predict_proba(X)[:, 1])
    return cbind(scores).T.corr()

def create_engine_from_config(config, section, prefix=None):
    '''
    Takes a config object returned by ConfigParser and returns an sqlalchemy
    enging object for the given database specified in the section parameter.
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

    a = import_module(features_file, folder_name)
    d = defaultdict(list)

    for i in dir(a):
        item = getattr(a, i)
        if callable(item):
            try:
                d[item.table_name].append(item)
            except:
                continue

    return d

class CategoricalImputer(TransformerMixin):
    '''
    Uses the training data to get the most common categories for each column.
    Then when transforming on new data, it makes sure to use the most common
    categories found in the training data to fill in missing values.
    '''

    def __init__(self, col):
        self.col = col

    def fit(self, X, y=None):
        self.val = X[self.col].apply(lambda x: x.value_counts().index[0])
        return self

    def transform(self, X):
        a = X[self.col].fillna(self.val)
        return cbind(X.drop(self.col, 1), a)

class OneHotEncode(TransformerMixin):
    '''
    Uses the training data to get all unique categories for each column and
    creates one dummy column for each unique category. Then when transforming
    on new data, it makes sure that the same dummy columns as found in the
    training data are created.
    '''

    def __init__(self, col):
        self.col = col

    def fit(self, X, y=None):
        X = pd.get_dummies(X[self.col], dummy_na=True)
        self.columns = X.columns
        return self

    def transform(self, X):
        Xdummy = pd.get_dummies(X[self.col], dummy_na=True)
        return cbind(X.drop(self.col, 1), Xdummy.T.reindex(self.columns).T.fillna(0))

def split_dir(directory, categories):
    files = os.listdir(directory)

    for cat in categories:
        cat_dir = os.path.join(directory, cat)
        os.makedirs(cat_dir)
        print 'folder created: %s' % cat_dir

        cat_files = [i for i in os.listdir(directory) if cat in i]
        for f in cat_files:
            shutil.move(os.path.join(directory, f), cat_dir)
        print 'files moved to folder: %s' % cat_dir

def train_valid_split(directory):
    train_dir = os.path.join(directory, 'train')
    valid_dir = os.path.join(directory, 'valid')
    os.makedirs(valid_dir)

    for cls in next(os.walk(train_dir))[1]:
        cls_dir = os.path.join(train_dir, cls)

        files = os.listdir(cls_dir)
        random.shuffle(files)
        valid_files = files[:int(len(files)*0.3)]
        valid_cls_dir = os.path.join(valid_dir, cls)
        os.makedirs(valid_cls_dir)
        print 'folder created: %s' % valid_cls_dir

        for f in valid_files:
            shutil.move(os.path.join(cls_dir, f), valid_cls_dir)
        print 'files moved to folder: %s' % valid_cls_dir

def sample_dir(directory):
    parts = directory.split('/')
    sample_dir = os.path.join(parts[0], 'sample', parts[1])
    os.makedirs(sample_dir)

    for cls in next(os.walk(directory))[1]:
        cls_dir = os.path.join(directory, cls)

        files = os.listdir(cls_dir)
        random.shuffle(files)
        sample_files = files[:100]

        sample_cls_dir = os.path.join(sample_dir, cls)
        os.makedirs(sample_cls_dir)
        print 'folder created: %s' % sample_cls_dir

        for f in sample_files:
            shutil.copy(os.path.join(cls_dir, f), sample_cls_dir)
        print 'files copied to folder: %s' % sample_cls_dir

def create_pred_csv(model, directory, batch_size):
    unknown_folder = os.path.join(directory, 'unknown')
    try:
        os.makedirs(unknown_folder)
    except OSError:
        if not os.path.isdir(unknown_folder):
            raise

    files = os.listdir(unknown_folder)
    num_files = len(files)
    generator = image.ImageDataGenerator()
    batches = generator.flow_from_directory(directory, target_size=(224,224), shuffle=False, class_mode=None,
                                            batch_size=batch_size)
    filenames = batches.filenames
    batches = itertools.islice(batches, 5)

    with open('predictions.csv', "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(('id', 'label'))

        for i, batch in enumerate(batches):
            if i >= num_files / batch_size:
                break

            ids = [name.split('/')[1].split('.')[0] for name in filenames[batch_size*i: batch_size*(i+1)]]
            preds = model.predict(batch)[1]
            data = zip(ids, preds)
            writer.writerows(data)

            print('Finished %s batches: %s images' % (i+1, (i+1)*batch_size))

def create_dir(folder):
    try:
        os.makedirs(folder)
    except OSError:
        if not os.path.isdir(folder):
            raise
