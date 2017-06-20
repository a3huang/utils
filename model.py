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
