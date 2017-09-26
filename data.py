from datetime import datetime
from pandas.tseries.offsets import *
from patsy import dmatrix

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

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

def time_unit(a, unit):
    '''
    Extract the value of a date variable with respect to a given time unit.

    ex) df['date'].pipe(time_unit, 'weekday')
    '''

    return getattr(a.dt, unit)

def qcut(a, q=10):
    '''
    Cut a continuous variable into quantiles with the smallest quantile equal
    to 1. Unlike pd.cut, allows for repeated bin edges.

    ex) df['HP'].pipe(qcut)
    '''

    a = a.sort_values().reset_index().reset_index()
    a['quantile'] = pd.qcut(a['level_0'], 10, labels=False) + 1
    return a.set_index('index').sort_index().reset_index()['quantile']

def top(a, n=None):
    '''
    Keep the top n most common levels of a categorical variable and label the
    rest as 'other'.

    ex) df['Type'].pipe(top, 5)
    '''

    if n:
        counts = a.fillna('missing').value_counts()
        top = counts.iloc[:n].index
        return a.apply(lambda x: x if x in top else 'other')
    else:
        return a

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

def count_missing(df):
    '''
    Count the number of missing values in each column.

    ex) df.pipe(count_missing).iloc[:5].sort_values().plot.barh()
    '''

    return df.isnull().sum()

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

def feature_scores(model, X, attr, sort_abs=False, top=None):
    '''
    Calculate importance scores for each feature for a given model.

    ex) feature_scores(rf, X_train, feature_importances_)
    '''

    if 'pipeline' in str(model.__class__):
        model = model.steps[-1][1]

    scores = getattr(model, attr)
    if len(scores.shape) > 1:
        scores = scores[0]

    df = pd.DataFrame(zip(X.columns, scores))

    if sort_abs:
        df['abs'] = np.abs(df[1])
        df = df.sort_values(by='abs', ascending=False).drop('abs', 1)
    else:
        df = df.sort_values(by=1, ascending=False)

    if top:
        return df[:top]
    else:
        return df

def top_corr(df, n=None):
    '''
    Show the top correlated pairs of variables sorted by magnitude.

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

def scoring_table(scores):
    '''
    Takes a dataframe containing predicted scores for a binary classification
    problem in the 1st column and true labels in the 2nd column. Creates an
    aggregated scoring table based on deciles.

    ex) scoring_table(cbind(model.predict_proba(X_test)[:, 1], y_test))
    '''

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

    df['KS'] = df['cumulataive_1'] - df['cumulative_0']
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

def compare_datasets_test(model, datasets, target, omit=None, threshold=0.5, random_state=42):
    '''
    Compares the AUC, confusion matrix, and classification report (precision,
    recall, f1 score) for a given model over several datasets.

    ex) compare_datasets_test(model, [df1, df2, df3, df4, df5], target='cancel',
            omit=['user_id'], threshold=0.1)
    '''

    for df in datasets:
        X = df.drop(omit + [target], 1)
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

def compare_datasets_cv(model, datasets, target, omit=None, random_state=42):
    '''
    Compares mean 5-fold CV AUC for a given model over several datasets.

    ex) compare_datasets_cv(model, [df1, df2, df3, df4, df5], target='cancel',
            omit=['user_id'])
    '''

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    for df in datasets:
        X = df.drop(omit + [target], 1)
        y = df[target]
        print cross_val_score(model, X, y, cv=cv, scoring='roc_auc').mean()

def compare_models_cv(models, df, target, omit=None, random_state=42):
    '''
    Compares mean 5-fold CV AUC for a given dataset over several models.

    ex) compare_models_cv([model1, model2, model3], df, target='cancel',
            omit=['user_id'])
    '''

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    X = df.drop(omit + [target], 1)
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

    scores = []
    for i, df in enumerate(datasets):
        scores_by_df = []
        for model in models:
            X = df.drop(omit + [target], 1)
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

def cbind2(df, obj, **kwargs):
    '''
    Append a column or dataframe to an existing dataframe as new columns.

    ex) df.pipe(cbind, a)
    '''

    objects = [df.reset_index(drop=True), pd.DataFrame(obj).reset_index(drop=True)]
    return pd.concat(objects, axis=1, **kwargs)

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

def interaction(df, col1, col2):
    '''
    Create interaction terms between 2 variables.

    ex) df.pipe(interaction, col1, col2)
    '''

    formula = '%s:%s - 1' % (col1, col2)
    X = dmatrix(formula, df)
    return pd.DataFrame(X, columns=X.design_info.column_names)

def get_index(df, col_names):
    return [df.columns.get_loc(i) for i in col_names]

def mark_confusion_errors(model, X, y, threshold=0.5):
    target = pd.DataFrame(y)
    target.columns = ['target']

    prediction = pd.DataFrame(model.predict_proba(X)[:, 1] > threshold, columns=['prediction'])

    df = cbind([X, target, prediction])
    df.loc[(df['prediction'] == 1) & (df['target'] == 0), 'error'] = 'FP'
    df.loc[(df['prediction'] == 0) & (df['target'] == 1), 'error'] = 'FN'
    df.loc[(df['prediction'] == df['target']), 'error'] = 'Correct'
    df = df.fillna(0)
    return df

def compare_roc_curves(model, datasets, target, omit=None, threshold=0.5, random_state=42):
    '''
    Compares the ROC curves for a given model on a fixed test set of the data.

    ex) compare_roc_curves(model, [df1, df2, df3, df4, df5], target='cancel',
            omit=['user_id'], threshold=0.1)
    '''

    for df in datasets:
        X = df.drop(omit + [target], 1)
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
            random_state=random_state)

        model.fit(X_train, y_train)

        pred = model.predict_proba(X_test)[:, 1]
        true = y_test

        fpr, tpr, _ = roc_curve(true, pred)
        plt.plot(fpr, tpr)

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

def missing_ind(a):
    return a.isnull()

def undummy_set(df, columns, name):
    a = cbind([df[df.columns.difference(columns)], df[columns].pipe(undummy)])
    a = a.rename(columns={0: name})
    return a

def binarize(a):
    return a.apply(lambda x: 1 if x > 0 else 0)
