import itertools
import os
import pandas as pd
import numpy as np

from collections import defaultdict, OrderedDict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import lime.lime_tabular
import matplotlib.pyplot as plt

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
        subsets = df.columns

    if isinstance(subsets[0], (tuple, list)):
        for cols in subsets:
            for i, j in itertools.combinations(cols, 2):
                df['%s*%s' % (i, j)] = df[i] * df[j]
    else:
        for i, j in itertools.combinations(subsets, 2):
            df['%s*%s' % (i, j)] = df[i] * df[j]
    return df

def sorted_fi(df, scores, top=10):
    return sorted(zip(df.columns, scores), key=lambda x: abs(x[1]), reverse=True)[:top]

# test if pipeline part actually works
def _get_feature_importances(model):
    if 'pipeline' in repr(model.__class__):
        model = model.steps[-1][1]

    for i in ['coef_', 'feature_importances_', 'ranking_', 'scores_']:
        if hasattr(model, i):
            scores = getattr(model, i)

            if len(scores.shape) > 1:
                scores = scores[0]
        return scores

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

def evaluate_feature_sets(model, dfs):
    l = []
    for X, y in dfs:
        #cv = StratifiedKFold(n_splits=5, shuffle=True)

        auc = cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()
        recall = cross_val_score(model, X, y, cv=5, scoring=decile_recall).mean()

        l.append((auc, recall))

    return pd.DataFrame(l)

def evaluate_feature_sets_boot(model, dfs, B=100):
    l = []
    for X, y in dfs:
        l1 = []
        df = pd.concat([X, y], axis=1)
        for i in range(B):
            df_b = df.sample(len(df), replace=True)

            X = df_b.iloc[:, :-1]
            y = df_b.iloc[:, -1]

            model.fit(X, y)

            df_test = df.loc[~df.index.isin(df_b.index)]
            X_test = df_test.iloc[:, :-1]
            y_test = df_test.iloc[:, -1]

            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            recall = decile_recall(model, X_test, y_test)

            l1.append((auc, recall))

        l1 = pd.DataFrame(l1)
        l.append(l1)
        #l1 = np.array(l1)
        #l.append((np.mean(l1[:, 0]), np.mean(l1[:, 1])))

    return pd.concat(l, axis=1)
    #return pd.DataFrame(l)

def evaluate_rocs(model, dfs):
    l = []
    for i, df in enumerate(dfs):
        X = df[0]
        y = df[1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model.fit(X_train, y_train)

        try:
            prediction = model.predict_proba(X_test)[:, 1]
        except:
            prediction = model.predict(X_test)

        fpr, tpr, _ = roc_curve(y_test, prediction)
        plt.plot(fpr, tpr, label='Model %s' % i)

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

# needs pred column
def get_error_dfs(df, target):
    df = df.copy()
    df.loc[(df[target] == False) & (df['pred'] == True), 'fp']  = 1
    df.loc[df[target] == df['pred'], 'correct'] = 1
    df.loc[(df[target] == True) & (df['pred'] == False), 'fn'] = 1
    df = df.fillna(0)
    return df

def create_explainer(model, X_train):
    mi = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns.values)
    return mi

def plot_explanations(mi, model, X_test, i=None):
    if i is None:
        i = np.random.randint(0, X_test.shape[0])
    exp = mi.explain_instance(X_test.values[i], model.predict_proba)
    a = pd.DataFrame(exp.as_list()).sort_index(ascending=False).set_index(0)
    colors = ''.join(['r' if i >= 0 else 'g' for i in a[1]])
    a.plot.barh(color=colors)
    plt.legend().remove()

def feat_shuffle(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model.fit(X_train, y_train)
    auc_original = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    recall_original = decile_recall(model, X_test, y_test)

    d = defaultdict(list)
    for col in X_train.columns:
        auc_list = []
        recall_list = []
        for i in range(10):
            X_train.loc[:, col] = np.random.permutation(X_train.loc[:, col])
            #X_train.loc[:, col] = X_train.loc[:, col].sample(X_train.shape[0], replace=True)
            model.fit(X_train, y_train)

            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            recall = decile_recall(model, X_test, y_test)

            auc_list.append(abs(auc - auc_original))
            recall_list.append(abs(recall - recall_original))

        d['variable'].extend([col]*10)
        d['auc'].extend(auc_list)
        d['recall'].extend(recall_list)

    return pd.DataFrame(d)

def plot_fs_boxplots(df, score, top=None, **kwargs):
    order = df[['variable']].pipe(add_column, df.groupby('variable').transform(lambda x: x.max() - x.min()))
    order = order.sort_values(by=score, ascending=False).groupby('variable').head(1)['variable']

    if top:
        order = order[:top]
        df = df[df['variable'].isin(order)]

    df.pipe(plot_box, 'variable', score, top=top, order=order, **kwargs)

def get_top_n(model, X_train, y_train, n, coeff=True):
    model.fit(X_train, y_train)

    if coeff:
        top = sorted_fi(X_train, _get_feature_importances(model), top=n)
        return top
    else:
        return [i[0] for i in top]
