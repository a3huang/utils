import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sklearn import tree
import pydotplus
import subprocess

from data import crosstab
from model import _get_feature_importances, _get_model_name

# Helper Functions
def top_n_cat(a, n=5):
    a = a.fillna('missing')
    counts = a.value_counts()
    top = counts.iloc[:n].index
    return a.apply(lambda x: x if x in top else 'other')

def bin_cont(a, n=5):
    if a.value_counts().shape[0] <= n:
        return a

    num_bins = int(np.ceil((a.max() - a.min())/n))
    min_edge = np.floor(a.min()/10)
    bin_edges = [min_edge + n*i for i in range(num_bins+2)]
    return pd.cut(a, bins=bin_edges, include_lowest=True)

def treat(a, n=5):
    if a.dtype == 'O':
        return top_n_cat(a, n)
    elif a.dtype in ['int64', 'float64']:
        return bin_cont(a, n)

def winsorize(x, p=.05):
    n = int(1/p)
    sorted_col = x.sort_values().reset_index(drop=True)
    quantiles = pd.qcut(sorted_col.reset_index()['index'], n).cat.codes
    a = pd.concat([sorted_col, quantiles], axis=1)
    quantiles_to_keep = a[0].unique()[1:-1]
    return a[a[0].isin(quantiles_to_keep)].iloc[:, 0]

# Main Functions
def plot_missing(df, top=None, **kwargs):
    a = df.isnull().mean(axis=0)
    a = a[a > 0]

    if len(a) == 0:
        return 'No Missing Values'

    a = a.sort_values(ascending=False)

    if top:
        a = a[:top]

    a.sort_values().plot.barh(**kwargs)
    plt.xlabel('Proportion')
    plt.title('Missing')
    return a

def plot_bar(df, *args, **kwargs):
    if len(args) == 1:
        if isinstance(args[0], list):
            return _plot_bar_col_multi(df, *args, **kwargs)
        else:
            return _plot_bar_col(df, *args, **kwargs)

    elif len(args) == 2:
        if isinstance(args[1], list):
            return _plot_bar_col_multi_groupby_cat(df, *args, **kwargs)
        else:
            return _plot_bar_col_groupby_cat(df, *args, **kwargs)

    elif len(args) == 3:
        return _plot_bar_col_groupby_cat2(df, *args, **kwargs)

    else:
        raise ValueError, 'Not a valid number of arguments'

def _plot_bar_col(df, col, top=20, **kwargs):
    # if col.dtype == 'O':
    #   top_n_cat(col)
    # if col.dtype in ['int64', 'float64']:
    #   bin(col)

    a = top_n_cat(df[col], top)

    a = a.value_counts(dropna=False)
    a = a / float(sum(a))

    a.sort_values().plot.barh(**kwargs)
    plt.xlabel('Proportion')
    plt.title(col)
    return a

def _plot_bar_col_multi(df, col, top=20, **kwargs):
    # only supports column indices
    a = df.iloc[:, col].sum().sort_values(ascending=False)
    a = a / float(sum(a))

    a.sort_values().plot.barh(**kwargs)
    plt.xlabel('Proportion')
    return a

def _plot_bar_col_groupby_cat(df, cat, col, as_cat=False, top=20, **kwargs):
    df = df.copy()
    df[cat] = top_n_cat(df[cat], top)

    if as_cat or df[col].dtype == 'O':
        df[col] = top_n_cat(df[col], top)
        a = pd.crosstab(df[cat], df[col], normalize='index')
        a.plot.barh(**kwargs)
        plt.gca().invert_yaxis()
        plt.xlabel('Proportion')
        plt.title('%s grouped by %s' % (col, cat))
        plt.legend(title=col, loc=(1, 0.5))
    else:
        a = df.groupby(cat)[col].mean()
        a.sort_index(ascending=False).plot.barh(**kwargs)
        plt.xlabel('Mean')
        plt.title('%s grouped by %s' % (col, cat))

    return a

def _plot_bar_col_multi_groupby_cat(df, cat, col_list, as_cat=False, top=20, **kwargs):
    # df = df.copy()
    # df[cat] = top_n_cat(df[cat], top)
    #
    # a = df.groupby('cat')[col_list]
    # a.plot.barh(**kwargs)
    # plt.gca().invert_yaxis()
    # plt.xlabel('Mean')
    # plt.legend(title=cat, loc=(1, 0.5))

    a = df.melt([cat], col_list)
    return _plot_bar_col_groupby_cat2(a, 'variable', cat, 'value', **kwargs)

def _plot_bar_col_groupby_cat2(df, cat1, cat2, col, top=20, **kwargs):
    df = df.copy()
    df[cat1] = top_n_cat(df[cat1], top)
    df[cat2] = top_n_cat(df[cat2], top)

    a = pd.crosstab(df[cat1], df[cat2], df[col], aggfunc=np.mean)
    a.plot.barh(**kwargs)
    plt.gca().invert_yaxis()
    plt.xlabel('Mean')
    plt.title('%s grouped by %s and %s' % (col, cat1, cat2))
    plt.legend(title=cat2, loc=(1, 0.5))
    return a


def plot_box(df, *args, **kwargs):
    if len(args) == 2:
        if isinstance(args[1], list):
            return _plot_box_col_multi_groupby_cat(df, *args, **kwargs)
        else:
            return _plot_box_col_groupby_cat(df, *args, **kwargs)

    else:
        raise ValueError, 'Not a valid number of arguments'

def _plot_box_col_groupby_cat(df, cat, col, showfliers=False, top=20, **kwargs):
    df = df.copy()
    df[cat] = top_n_cat(df[cat], top)

    sns.boxplot(y=cat, x=col, data=df, showfliers=showfliers, orient='h', **kwargs)
    plt.xlabel(col)
    plt.ylabel(cat)
    plt.title('%s grouped by %s' % (col, cat))

def _plot_box_col_multi_groupby_cat(df, cat, col_list, showfliers=False, top=20, **kwargs):
    df = df.copy()
    df[cat] = top_n_cat(df[cat], top)

    a = df.melt([cat], col_list)
    sns.boxplot(y='variable', x='value', hue=cat, data=a, showfliers=showfliers,
                **kwargs)
    plt.legend(loc=(1, .5))


def plot_heatmap(df, *args, **kwargs):
    if len(args) == 2:
        return _plot_heatmap_groupby_cat2(df, *args, **kwargs)

    elif len(args) == 3:
        return _plot_heatmap_col_groupby_cat2(df, *args, **kwargs)

    else:
        raise ValueError, 'Not a valid number of arguments'

def _plot_heatmap_groupby_cat2(df, cat1, cat2, normalize='index', top=20, **kwargs):
    df = df.copy()
    df[cat1] = top_n_cat(df[cat1], top)
    df[cat2] = top_n_cat(df[cat2], top)

    a = pd.crosstab(df[cat1], df[cat2], normalize=normalize)
    sns.heatmap(a, annot=True, fmt='.2f', **kwargs)
    plt.gca().invert_yaxis()
    plt.title('%s and %s' % (cat1, cat2))
    return a

def _plot_heatmap_col_groupby_cat2(df, cat1, cat2, col, top=20, **kwargs):
    df = df.copy()
    df[cat1] = top_n_cat(df[cat1], top)
    df[cat2] = top_n_cat(df[cat2], top)

    a = pd.crosstab(df[cat1], df[cat2], df[col], aggfunc=np.mean)
    sns.heatmap(a, annot=True, fmt='.2f', **kwargs)
    plt.gca().invert_yaxis()
    plt.title('%s grouped by %s and %s' % (col, cat1, cat2))
    return a


def plot_hist(df, *args, **kwargs):
    if len(args) == 1:
        return _plot_hist_col(df, *args, **kwargs)

    elif len(args) == 2:
        return _plot_hist_col_groupby_cat(df, *args, **kwargs)

    else:
        raise ValueError, 'Not a valid number of arguments'

def _plot_hist_col(df, col, density=False, winsorize_col=True, **kwargs):
    a = df[col]

    if winsorize_col:
        a = winsorize(a)

    assert len(a.shape) == 1, 'Must be single column'
    assert len(a) > 0, 'Must have at least 1 element'

    if density:
        kde = True
        hist = False
    else:
        kde = False
        hist = True

    weights = np.ones_like(a) / float(len(a))
    sns.distplot(a, hist=hist, kde=kde, **kwargs)
    plt.ylabel('Proportion')
    plt.title(col)

def _plot_hist_col_groupby_cat(df, cat, col, density=False, winsorize_col=True, top=20, **kwargs):
    df = df.copy()
    df[cat] = top_n_cat(df[cat], top)

    if winsorize_col:
        df[col] = winsorize(df[col])
        df = df[~df[col].isnull()]

    assert df[col].isnull().any() == False, 'Column contains null values'

    if density:
        kde = True
        hist = False
    else:
        kde = False
        hist = True

    bins = np.histogram(df[col])[1]
    groups = df.groupby(cat)[col]
    for k, v in groups:
        sns.distplot(v, hist=hist, kde=kde, bins=bins, label=str(k), **kwargs)

    plt.legend(title=cat, loc=(1, 0.5))
    plt.title(col)

def plot_line(df, *args, **kwargs):
    if len(args) == 2:
        return _plot_line_col_groupby_cat(df, *args, **kwargs)

    elif len(args) == 3:
        return _plot_line_col_groupby_cat2(df, *args, **kwargs)

    else:
        raise ValueError, 'Not a valid number of arguments'

def _plot_line_col_groupby_cat(df, cat, col, top=20, **kwargs):
    df = df.copy()
    df[cat] = top_n_cat(df[cat], top)

    a = df.groupby(cat)[col].mean()
    a.plot(**kwargs)
    plt.xlabel(cat)
    plt.ylabel('Mean')
    plt.title('%s grouped by %s' % (col, cat))
    return a

def _plot_line_col_groupby_cat2(df, cat1, cat2, col, top=20, **kwargs):
    df = df.copy()
    df[cat1] = top_n_cat(df[cat1], top)
    df[cat2] = top_n_cat(df[cat2], top)

    a = pd.crosstab(df[cat1], df[cat2], df[col], aggfunc=np.mean)
    a.plot(**kwargs)
    plt.xlabel(cat1)
    plt.ylabel('Mean of %s' % col)
    plt.title('Interaction Effect of %s and %s on %s' % (cat1, cat2, col))
    plt.legend(title=cat2, loc=(1, 0.5))
    return a


def plot_scatter(df, *args, **kwargs):
    if len(args) == 2:
        return _plot_scatter_col2(df, *args, **kwargs)

    elif len(args) == 3:
        return _plot_scatter_col2_groupby_cat(df, *args, **kwargs)

    else:
        raise ValueError, 'Not a valid number of arguments'

def _plot_scatter_col2(df, col1, col2, **kwargs):
    sns.lmplot(col1, col2, data=df, ci=False, **kwargs)

def _plot_scatter_col2_groupby_cat(df, cat, col1, col2, top=20, **kwargs):
    df = df.copy()
    df[cat] = top_n_cat(df[cat], top)

    sns.lmplot(col1, col2, hue=cat, data=df, fit_reg=False, **kwargs)


def plot_ts_line(df, cat=None, col=None, **kwargs):
    if cat is None and col is None:
        return _plot_ts_counts(df, kind='line', **kwargs)

    elif cat is None:
        return _plot_ts_col(df, col, kind='line', **kwargs)

    elif col is None:
        return _plot_ts_counts_groupby_cat(df, cat, kind='line', **kwargs)

    else:
        return _plot_ts_col_groupby_cat(df, cat, col, kind='line', **kwargs)

def plot_ts_area(df, cat=None, col=None, **kwargs):
    if cat is None and col is None:
        return _plot_ts_counts(df, kind='area', **kwargs)

    elif cat is None:
        return _plot_ts_col(df, col, kind='area', **kwargs)

    elif col is None:
        return _plot_ts_counts_groupby_cat(df, cat, kind='area', **kwargs)

    else:
        return _plot_ts_col_groupby_cat(df, cat, col, kind='area', **kwargs)

def plot_ts_bar(df, cat=None, col=None, **kwargs):
    if cat is None and col is None:
        return _plot_ts_counts(df, kind='bar', **kwargs)

    elif cat is None:
        return _plot_ts_col(df, col, kind='bar', **kwargs)

    elif col is None:
        return _plot_ts_counts_groupby_cat(df, cat, kind='bar', **kwargs)

    else:
        return _plot_ts_col_groupby_cat(df, cat, col, kind='bar', **kwargs)

def _plot_ts_counts(df, kind, date_col='date', freq='M', **kwargs):
    df = df.copy()

    if freq in ['hour', 'month', 'weekday']:
        df[date_col] = getattr(df.set_index(date_col).index, freq)
        grouper = df.groupby(date_col)
    else:
        grouper = df.set_index(date_col).resample(freq)

    grouper.size().plot(kind=kind, **kwargs)

def _plot_ts_col(df, col, kind, date_col='date', freq='M', **kwargs):
    df = df.copy()

    if freq in ['hour', 'month', 'weekday']:
        df[date_col] = getattr(df.set_index(date_col).index, freq)
        grouper = df.groupby(date_col)
    else:
        grouper = df.set_index(date_col).resample(freq)

    grouper[col].mean().plot(kind=kind, **kwargs)
    plt.legend(loc=(1, 0.5))

def _plot_ts_counts_groupby_cat(df, cat, kind, date_col='date', freq='M',
                                top=20, **kwargs):
    df = df.copy()
    df[cat] = top_n_cat(df[cat], top)

    if freq in ['hour', 'month', 'weekday']:
        df[date_col] = getattr(df.set_index(date_col).index, freq)
    else:
        df = df.set_index(date_col).to_period(freq).reset_index()

    a = df.pipe(crosstab, date_col, cat)
    a.plot(kind=kind, **kwargs)
    plt.legend(loc=(1, 0.5))

    return a

def _plot_ts_col_groupby_cat(df, cat, col, kind, date_col='date', freq='M',
                             top=20, **kwargs):
    df = df.copy()
    df[cat] = top_n_cat(df[cat], top)

    if freq in ['hour', 'month', 'weekday']:
        df[date_col] = getattr(df.set_index(date_col).index, freq)
    else:
        df = df.set_index(date_col).to_period(freq).reset_index()

    a = df.pipe(crosstab, date_col, cat, col)
    a.plot(kind=kind, **kwargs)
    plt.legend(loc=(1, 0.5))

    return a


def plot_ts_box(df, col, date_col='date', freq='M', **kwargs):
    df = df.copy()

    if freq in ['month', 'weekday', 'hour']:
        df[date_col] = getattr(df.set_index(date_col).index, freq)
    else:
        df = df.set_index(date_col).to_period(freq)

    df.boxplot(by=date_col, column=col, **kwargs)
    plt.xticks(rotation=90)
########


def plot_pca(df, cat, pca_model=None, sample_size=1000, **kwargs):
    df = df.copy()
    s = StandardScaler()

    if pca_model is None:
        pca_model = PCA()

    if sample_size:
        df = df.sample(sample_size)

    X = df[df.columns.difference([cat])]
    df['PCA 1'] = pca_model.fit_transform(s.fit_transform(X))[:, 0]
    df['PCA 2'] = pca_model.fit_transform(s.fit_transform(X))[:, 1]
    plot_scatter_groupby_1(df, cat, 'PCA 1', 'PCA 2', **kwargs)

def plot_clusters(df, cluster_model=None, pca_model=None, sample_size=1000, **kwargs):
    df = df.copy()
    s = StandardScaler()

    if cluster_model is None:
        cluster_model = KMeans()

    if sample_size:
        df = df.sample(sample_size)

    cluster_model.fit(s.fit_transform(df))
    df['cluster'] = cluster_model.labels_
    plot_pca(df, 'cluster', pca_model, sample_size=None, **kwargs)

def plot_decision_tree(df, target, filename, **kwargs):
    X = df[df.columns.difference([target])]
    y = df[target]

    model = tree.DecisionTreeClassifier(**kwargs)
    model.fit(X, y)
    dot_data = tree.export_graphviz(model,
                                    out_file=None,
                                    feature_names=X.columns,
                                    filled=True,
                                    class_names=y.astype(str).unique())
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(filename)
    subprocess.call(('open', filename))
    return model

def plot_feature_importances(df, model, target, top=None, **kwargs):
    X = df[df.columns.difference([target])]
    a = _get_feature_importances(model, X)
    model_name = _get_model_name(model)

    if top:
        a = a[:top]

    a.sort_index(ascending=False).set_index(0).plot.barh(**kwargs)
    plt.ylabel('Feature')
    plt.title(model_name)
    plt.legend().remove()
    return a

def plot_correlation_matrix(df, **kwargs):
    a = df.corr()
    sns.heatmap(a, annot=True, fmt='.2f', **kwargs)
    return a

def plot_confusion_matrix(df, model, target, threshold=0.5, **kwargs):
    X = df.drop(target, 1)
    y = df[target]

    try:
        prediction = model.predict_proba(X)[:, 1] > threshold
    except:
        prediction = model.predict(X)

    a = confusion_matrix(y, prediction)
    a = a / float(sum(sum(a)))
    sns.heatmap(a, annot=True, fmt='.2f', **kwargs)
    return a

def plot_roc_curve(df, model, target):
    X = df.drop(target, 1)
    y = df[target]

    model_name = _get_model_name(model)

    try:
        prediction = model.predict_proba(X)[:, 1]
    except:
        prediction = model.predict(X)

    fpr, tpr, _ = roc_curve(y, prediction)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(model_name)

def plot_learning_curves(df, model, target):
    X = df.drop(target, 1)
    y = df[target]

    cv = StratifiedKFold(n_splits=5, shuffle=True)

    model_name = _get_model_name(model)

    sizes, train, validation = learning_curve(model, X, y, cv=cv, scoring='roc_auc')
    plt.plot(sizes, np.mean(train, axis=1), label='train')
    plt.plot(sizes, np.mean(validation, axis=1), label='validation')
    plt.xlabel('Sample Size')
    plt.ylabel('Performance')
    plt.title(model_name)
    plt.legend(loc=(1, 0.5))

def plot_prob_estimates(df, model):
    model_name = _get_model_name(model)

    plt.hist(model.predict_proba(df)[:, 1])
    plt.label('Probability')
    plt.title(model_name)

def plot_word_frequencies(docs, top=20, **kwargs):
    c = CountVectorizer()
    c.fit(docs)

    counts = c.fit_transform(docs).sum(axis=0)
    counts = np.array(counts).squeeze()

    vocab = pd.DataFrame([(k, counts[v]) for k, v in c.vocabulary_.items()], columns=['Word', 'Count'])
    total = vocab['Count'].sum()
    vocab['Frequency'] = vocab['Count'] / total
    vocab = vocab.sort_values(by='Count', ascending=False).set_index('Word')

    if top:
        vocab = vocab[:top]

    vocab[['Frequency']].sort_values(by='Frequency').plot.barh(**kwargs)

    plt.xlabel('Proportion')
    plt.ylabel('Words')
    plt.legend().remove()

    return vocab
