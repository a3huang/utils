import pandas as pd

from collections import OrderedDict
from sklearn.model_selection import StratifiedKFold

# how to handle classifiers with multiple classes?
def _get_feature_importances(model):
    try:
        return model.coef_[0]
    except:
        try:
            return model.feature_importances_
        except:
            return model.scores_

def _get_top_n_features(model, X):
    if 'sequentialfeatureselector' in repr(model.__class__).lower():
        col = X.columns[list(model.k_feature_idx_)]
    else:
        a = sorted(zip(X.columns, get_feature_importances(model)), key=lambda x: abs(x[1]), reverse=True)[:10]
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
    if not os.path.exists(folder):
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
        model_name = get_model_name(model)

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
        l.extend(sorted(zip(X_train.columns, _get_feature_importances(model)), key=lambda x: abs(x[1]), reverse=True)[:10])

    feat = pd.DataFrame(l, columns=['features', 'scores'])
    feat_props = feat.groupby('features').size() / (len(models) * 10.0)
    return feat_props.sort_values(ascending=False)

def feature_selection_stability(X, y, feat_model, model):
    cols_selected = []
    cv_score = []

    cv = StratifiedKFold(n_splits=5, shuffle=True)

    for tr, te in cv.split(X, y):
        feat_model.fit(X.values[tr], y.values[tr])
        col = get_top_n_features(feat_model, X)
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

    return np.mean(cv_score), len(inter) / float(len(union))
