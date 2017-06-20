import pandas as pd

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

# factor out the sorted zip thing?
# models = [LogisticRegression(), RandomForestClassifier(),
#           SelectKBest(score_func=mutual_info_classif), LinearSVC()]
# feature_selection_suite(X_train, y_train, models)
def feature_selection_suite(X, y, models):
    l = []
    for model in models:
        model.fit(X, y)
        l.extend(sorted(zip(X_train.columns, _get_feature_importances(model)), key=lambda x: abs(x[1]), reverse=True)[:10])

    feat = pd.DataFrame(l, columns=['features', 'scores'])
    feat_props = feat.groupby('features').size() / (len(models) * 10.0)
    return feat_props.sort_values(ascending=False)

# feat_select_stability(X_train, y_train, RandomForestClassifier(), LogisticRegression())
def feature_selection_stability(X, y, feat_model, model):
    cols_selected = []
    cv_score = []

    cv = StratifiedKFold(n_splits=5, shuffle=True)
    for tr, te in cv.split(X, y):
        feat_model.fit(X.values[tr], y.values[tr])
        a = sorted(zip(X.columns, _get_feature_importances(feat_model)), key=lambda x: abs(x[1]), reverse=True)[:10]

        col = [i[0] for i in a]
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
