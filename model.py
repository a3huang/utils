import itertools
import os
import pandas as pd
import numpy as np

from collections import OrderedDict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class Transform(BaseEstimator, TransformerMixin):
    def __init__(self, d):
        self.d = d

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col, f in self.d.items():
            if isinstance(col, tuple):
                col = list(col)
            X = np.apply_along_axis(f, 0, X)
        return X

class Interact(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X = pd.DataFrame(X)
        return interactions(X, self.col)

def interactions(df, subsets=None):
    df = df.copy()
    if subsets is None:
        subsets = [df.columns]
    for cols in subsets:
        for i, j in itertools.combinations(cols, 2):
            df['%s*%s' % (i, j)] = df[i] * df[j]
    return df

# test if pipeline part actually works
def _get_feature_importances(model, X):
    if 'pipeline' in repr(model.__class__):
        model = model.steps[-1][1]

    for i in ['coef_', 'feature_importances_', 'ranking_', 'scores_']:
        if hasattr(model, i):
            scores = getattr(model, i)

            if len(scores.shape) > 1:
                scores = scores[0]

            a = pd.DataFrame(sorted(zip(X.columns, scores),
                key=lambda x: abs(x[1]), reverse=True))
            return a

def _get_top_n_features(model, X):
    if 'sequentialfeatureselector' in repr(model.__class__).lower():
        col = X.columns[list(model.k_feature_idx_)]
    else:
        a = sorted(zip(X.columns, _get_feature_importances(model)),
                key=lambda x: abs(x[1]), reverse=True)[:10]
        col = [i[0] for i in a]
    return col

def _get_model_name(model):
    if 'pipeline' in repr(model.__class__).lower():
        return model.steps[-1][0]
    else:
        return repr(model.__class__).split('.')[-1].split("'")[0].lower()

################################################################################

def decile_recall(model, X, y):
    scores = pd.concat([pd.DataFrame(model.predict_proba(X)[:, 1]),
        pd.DataFrame(y).reset_index(drop=True)], axis=1)
    return get_scoring_table(scores)['Target Metrics']['Cumulative'].loc[5]

def get_scoring_table(scores):
    scores.columns = ['scores', 'target']
    scores = scores.sort_values(by='scores', ascending=False).reset_index(drop=True)
    scores['Decile'] = pd.qcut(scores.index, 10, labels=False) + 1

    df = scores.groupby('Decile')['scores'].agg([min, max])
    df['obs'] = scores.groupby('Decile').size()
    df['comp'] = df['obs']/float(len(scores))
    df['cum'] = df['comp'].cumsum()
    df['obs_0'] = scores[scores['target'] == 0].groupby('Decile').size()
    df['comp_0'] = df['obs_0'] / float(len(scores[scores['target'] == 0]))
    df['cum_0'] = df['comp_0'].cumsum()
    df['obs_1'] = scores[scores['target'] == 1].groupby('Decile').size()
    df['comp_1'] = df['obs_1'] / float(len(scores[scores['target'] == 1]))
    df['cum_1'] = df['comp_1'].cumsum()
    df['KS'] = df['cum_1'] - df['cum_0']
    df['rate'] = df['obs_1']/df['obs']
    df['index'] = df['rate'] / (len(scores[scores['target'] == 1])/float(len(scores))) * 100
    df = df.round(2)

    top_columns = ['scores']*2 + ['Population Metrics']*3 + ['Non-Target Metrics']*3 + \
                  ['Target Metrics']*3 + ['Validation Metrics']*3
    bottom_columns = ['Min Score', 'Max Score', 'Count', 'Composition', 'Cumulative', 'Count',
                      'Composition', 'Cumulative', 'Count', 'Composition', 'Cumulative', 'K-S',
                      'Cancel Rate', 'Cancel Index']
    df.columns = pd.MultiIndex.from_tuples(zip(top_columns, bottom_columns))

    return df

# have separate folder for each experiment?
def fit_models(models, X, y, folder=None):
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    d = OrderedDict()
    for i, model in enumerate(models):
        model_name = _get_model_name(model)

        if model_name in d:
            model_name = model_name + '_%s' % i
        d[model_name] = model

        model.fit(X, y)

        if folder:
            pickle.dump(model, open(folder + '%s_model.pkl' % model_name, 'wb'))

    if folder:
        df = pd.concat([X, y], axis=1)
        pickle.dump(df, open(folder + 'data.pkl', 'wb'))

    return d

# what to do about last slash in folder name?
def load_models(folder):
    model_files = glob.glob(folder + '*_model.pkl')

    d = {}
    for model_file in model_files:
        model = pickle.load(open(model_file, 'rb'))
        model_name = model_file.split('/')[-1].split('_')[0]
        d[model_name] = model

    return d

# how to handle models without predict_proba method
def evaluate_models_cv(model_dict, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    models = model_dict.values()

    l = []
    for model_name, model in model_dict.items():
        auc = []
        recall = []

        auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc').mean()
        recall = cross_val_score(model, X, y, cv=cv, scoring=decile_recall).mean()

        l.append((model_name, auc, recall))

    df = pd.DataFrame(l)
    df.columns = ['Model', 'AUC', '5th Decile Recall']
    return df

def evaluate_models_test(model_dict, X, y):
    models = model_dict.values()

    l = []
    for model in models:
        model_name = _get_model_name(model)

        auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
        recall = decile_recall(model, X, y)

        l.append((model_name, auc, recall))

    df = pd.DataFrame(l)
    df.columns = ['Model', 'AUC', '5th Decile Recall']
    return df

# factor out the sorted zip thing?
def feature_selection_suite(X, y, models):
    l = []
    for model in models:
        model.fit(X, y)
        l.extend(sorted(zip(X.columns, _get_feature_importances(model)), key=lambda x: abs(x[1]), reverse=True)[:10])

    feat = pd.DataFrame(l, columns=['features', 'scores'])
    feat_props = feat.groupby('features').size() / (len(models) * 10.0)
    return feat_props.sort_values(ascending=False)

def feature_selection_stability(X, y, feat_model, model):
    cols_selected = []
    cv_score = []

    cv = StratifiedKFold(n_splits=5, shuffle=True)

    for tr, te in cv.split(X, y):
        feat_model.fit(X.values[tr], y.values[tr])
        col = _get_top_n_features(feat_model, X)
        cols_selected.append(col)

        model.fit(X.loc[:, col].values[tr], y.values[tr])
        true = y.values[te]

        try:
            pred = model.predict_proba(X.loc[:, col].values[te])[:, 1]
        except:
            pred = model.predict(X.loc[:, col].values[te])

        score = roc_auc_score(true, pred)
        cv_score.append(score)

    intersect = set(cols_selected[0])
    union = set()
    for i in cols_selected:
        intersect = intersect.intersection(set(i))
        union = union.union(set(i))

    return np.mean(cv_score), len(intersect) / float(len(union))

# gets dataframe at nth step of a pipeline
def get_step_n_pipe(pipeline, n, X):
    a = pipeline.steps[0][1].transform(X.values)
    for i in range(1, n):
        a = pipeline.steps[i][1].transform(a)
    return pd.DataFrame(a, columns=X.columns)

def evaluate_dfs(model_dict, df_list):
    result = []
    for df in df_list:
        df1 = df[df['start'] > '2016-07-18']

        X = df1.drop(['user_id', 'start', 'end', 'days', 'cancel'], 1)
        y = df1['cancel']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

        result.append(evaluate_models_cv(model_dict, X_train, y_train))

    return pd.concat(result)

def evaluate_dfs_test(model_dict, df_list):
    result = []
    for df in df_list:
        df1 = df[df['start'] > '2016-07-18']

        X = df1.drop(['user_id', 'start', 'end', 'days', 'cancel'], 1)
        y = df1['cancel']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

        result.append(evaluate_models_test(model_dict, X_test, y_test))

    return pd.concat(result)

def evaluate_feat_selection(model_dict, feat_models, X, y):
    l = []
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    for model_name, model in model_dict.items():
        for feat_model in feat_models:
            feat_model_name = _get_model_name(feat_model)
            if hasattr(feat_model, 'estimator'):
                feat_model_name += ' + %s' % _get_model_name(feat_model.estimator)
            feat_model.fit(X.values, y.values)

            col = _get_top_n_features(feat_model, X)

            auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc').mean()
            recall = cross_val_score(model, X, y, cv=cv, scoring=decile_recall).mean()

            l.append((feat_model_name, model_name, auc, recall))
    return pd.DataFrame(l)

def evaluate_interactions(model_dict, X, y, interactions):
    l = []
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    for model_name, model in model_dict.items():
        model = make_pipeline(StandardScaler(), Interact(interactions), model)
        auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc').mean()
        recall = cross_val_score(model, X, y, cv=cv, scoring=decile_recall).mean()

        l.append((model_name, auc, recall))
    return pd.DataFrame(l)

def evaluate_transforms(model_dict, X, y, transforms):
    l = []
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    for model_name, model in model_dict.items():
        model = make_pipeline(StandardScaler(), Transform(transforms), model)
        auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc').mean()
        recall = cross_val_score(model, X, y, cv=cv, scoring=decile_recall).mean()

        l.append((model_name, auc, recall))
    return pd.DataFrame(l)

# input list of lists of features to drop?
def evaluate_feature_sets(model, dfs):
    l = []
    for df in dfs:
        X = df.drop(['user_id', 'start', 'end', 'days', 'cancel'], 1)
        y = df['cancel']

        cv = StratifiedKFold(n_splits=5, shuffle=True)

        auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc').mean()
        recall = cross_val_score(model, X, y, cv=cv, scoring=decile_recall).mean()

        l.append((auc, recall))
    return pd.DataFrame(l)

# needs pred column
def get_error_dfs(df, target):
    df = df.copy()
    df.loc[(df[target] == False) & (df['pred'] == True), 'fp']  = 1
    df.loc[df[target] == df['pred'], 'correct'] = 1
    df.loc[(df[target] == True) & (df['pred'] == False), 'fn'] = 1
    df = df.fillna(0)
    return df
