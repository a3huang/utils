import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler

from sklearn import tree
import pydotplus
import subprocess

from model import _get_feature_importances, _get_model_name
from data import crosstab

def _index_to_name(df, col):
    if isinstance(col, int):
        col = df.columns[col]
    return col

# test this
def _top_n_cat(a, n=5):
    counts = a.value_counts(dropna=False)
    top = counts.iloc[:n].index
    return a.apply(lambda x: x if x in top else 'other')

# need to test
# need to accept 1 arg as well
def winsorize(x, p=.05):
    n = int(1/p)
    sorted_col = x.sort_values().reset_index(drop=True)
    quantiles = pd.qcut(sorted_col.reset_index()['index'], n).cat.codes
    a = pd.concat([sorted_col, quantiles], axis=1)
    quantiles_to_keep = a[0].unique()[1:-1]
    return a[a[0].isin(quantiles_to_keep)].iloc[:, 0]

################################################################################

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
        return plot_bar_single_column(df, *args, **kwargs)
    elif len(args) == 2:
        if isinstance(args[1], list):
            a = df.melt([args[0]], args[1])
            return plot_bar_groupby_2(a, 'variable', args[0], 'value', **kwargs)
        else:
            return plot_bar_groupby_1(df, *args, **kwargs)
    elif len(args) == 3:
        return plot_bar_groupby_2(df, *args, **kwargs)
    else:
        raise ValueError, 'Too many arguments'

def plot_bar_single_column(arg1, arg2=None, top=20, **kwargs):
    if arg2 is None:
        assert len(arg1.shape) == 1, 'If only one argument, then must be single column'
        a = arg1
        col = a.name
    else:
        df = arg1
        col = arg2
        a = df[col]

    a = _top_n_cat(a, top)
    a = a.value_counts(dropna=False).sort_index()
    a = a / float(sum(a))
    a.sort_index(ascending=False).plot.barh(**kwargs)
    plt.xlabel('Proportion')
    plt.title(col)
    return a

def plot_bar_groupby_1(df, cat, col, as_cat=False, top=20, **kwargs):
    df = df.copy()
    df[cat] = _top_n_cat(df[cat], top)

    if as_cat or df[col].dtype == 'O':
        df[col] = _top_n_cat(df[col], top)
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

def plot_bar_groupby_2(df, cat1, cat2, col, top=20, **kwargs):
    df = df.copy()
    df[cat1] = _top_n_cat(df[cat1], top)
    df[cat2] = _top_n_cat(df[cat2], top)

    a = pd.crosstab(df[cat1], df[cat2], df[col], aggfunc=np.mean)
    a.plot.barh(**kwargs)
    plt.gca().invert_yaxis()
    plt.xlabel('Mean')
    plt.title('%s grouped by %s and %s' % (col, cat1, cat2))
    plt.legend(title=cat2, loc=(1, 0.5))
    return a

def plot_heatmap_groupby_2(df, cat1, cat2, col, top=20, **kwargs):
    df = df.copy()
    df[cat1] = _top_n_cat(df[cat1], top)
    df[cat2] = _top_n_cat(df[cat2], top)

    a = pd.crosstab(df[cat1], df[cat2], df[col], aggfunc=np.mean)
    sns.heatmap(a, annot=True, fmt='.2f', **kwargs)
    plt.gca().invert_yaxis()
    plt.title('%s grouped by %s and %s' % (col, cat1, cat2))
    return a

def plot_line_groupby_1(df, cat, col, top=20, **kwargs):
    df = df.copy()
    df[cat] = _top_n_cat(df[cat], top)

    a = df.groupby(cat)[col].mean()
    a.plot(**kwargs)
    plt.xlabel(cat)
    plt.ylabel('Mean')
    plt.title('%s grouped by %s' % (col, cat))
    return a

def plot_line_groupby_2(df, cat1, cat2, col, top=20, **kwargs):
    df = df.copy()
    df[cat1] = _top_n_cat(df[cat1], top)
    df[cat2] = _top_n_cat(df[cat2], top)

    a = pd.crosstab(df[cat1], df[cat2], df[col], aggfunc=np.mean)
    a.plot(**kwargs)
    plt.xlabel(cat1)
    plt.ylabel('Mean %s' % col)
    plt.title('Interaction Effect of %s and %s on %s' % (cat1, cat2, col))
    plt.legend(title=cat2, loc=(1, 0.5))
    return a

def plot_hist(df, *args, **kwargs):
    if len(args) == 1:
        return plot_hist_single_column(df, *args, **kwargs)
    elif len(args) == 2:
        return plot_hist_groupby_1(df, *args, **kwargs)
    else:
        raise ValueError, 'Too many arguments'

def plot_hist_single_column(arg1, arg2=None, winsorize_col=True, **kwargs):
    if arg2 is None:
        assert len(arg1.shape) == 1, 'If only one argument, then must be single column'
        a = arg1
        col = a.name
    else:
        df = arg1
        col = arg2
        a = df[col]

    if winsorize_col:
        a = winsorize(a)

    assert len(a.shape) == 1, 'Must be single column'
    assert len(a) > 0, 'Must have at least 1 element'

    weights = np.ones_like(a) / float(len(a))
    a.plot.hist(weights=weights, **kwargs)
    plt.ylabel('Proportion')
    plt.title(col)

def plot_hist_groupby_1(df, cat, col, winsorize_col=True, top=20, **kwargs):
    df = df.copy()
    df[cat] = _top_n_cat(df[cat], top)

    if winsorize_col:
        df[col] = winsorize(df[col])
        df = df[~df[col].isnull()]

    assert df[col].isnull().any() == False, 'Column contains null values'

    bins = np.histogram(df[col])[1]
    groups = df.groupby(cat)[col]
    for k, v in groups:
        plot_hist_single_column(v, bins=bins, winsorize_col=False, label=str(k),
                                alpha=0.3, **kwargs)

    plt.legend(title=cat, loc=(1, 0.5))
    plt.title(col)

def plot_density_groupby_1(df, cat, col, winsorize_col=True, top=20, **kwargs):
    df = df.copy()
    df[cat] = _top_n_cat(df[cat], top)

    if winsorize_col:
        df[col] = winsorize(df[col])
        df = df[~df[col].isnull()]

    assert df[col].isnull().any() == False, 'Column contains null values'

    groups = df.groupby(cat)[col]
    for k, v in groups:
        sns.kdeplot(v, shade=True, label=str(k), **kwargs)

    plt.legend(title=cat, loc=(1, 0.5))
    plt.title(col)

def plot_box_groupby_1(df, cat, col, top=20, **kwargs):
    df = df.copy()
    df[cat] = _top_n_cat(df[cat], top)

    df.boxplot(by=cat, column=col, vert=False, **kwargs)
    plt.gca().invert_yaxis()
    plt.xlabel(col)
    plt.ylabel(cat)
    plt.suptitle('')
    plt.title('%s grouped by %s' % (col, cat))

def plot_scatter(df, col1, col2, **kwargs):
    sns.lmplot(col1, col2, data=df, **kwargs)

def plot_scatter_groupby_1(df, cat, col1, col2, top=20, **kwargs):
    df = df.copy()
    df[cat] = _top_n_cat(df[cat], top)

    sns.lmplot(col1, col2, hue=cat, data=df, fit_reg=False, **kwargs)
    plt.title('%s vs. %s' % (col1, col2))

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

def plot_feature_imps(df, model, target, top=None, **kwargs):
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

def plot_word_freqs(docs, top=20, **kwargs):
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

# train, test = train_test_split(df, random_state=42)
# test.pipe(plot_feature_importances, model, 'cancel')
# model.fit(train.drop('Generation',1), train['Generation'])
##########




# needs column named date
# need to better format x axis labels
def plot_ts(df, col=None, freq='M'):
    df = df.copy()

    if freq in ['month', 'dow', 'hour']:
        df['date'] = getattr(df.set_index('date').index, freq)
        grouper = df.groupby('date')
    else:
        grouper = df.set_index('date').resample(freq)

    if col:
        grouper[col].mean().plot()
    else:
        grouper.size().plot()

# can combine with plot_ts function?
# is there any need to convert to dateindex first?
# a['date'].dt.to_period('M')
def plot_grouped_ts(df, cat, col=None, freq='M'):
    df = df.copy()

    if freq in ['month', 'dow', 'hour']:
        df['date'] = getattr(df.set_index('date').index, freq)
    else:
        df = df.set_index('date').to_period(freq).reset_index()

    if col:
        df.pipe(crosstab, 'date', cat, col).plot()
    else:
        df.pipe(crosstab, 'date', cat).plot()

# needs column named date
# need to better format x axis labels
def plot_ts_bar(df, col=None, freq='M'):
    df = df.copy()

    if freq in ['month', 'dow', 'hour']:
        df['date'] = getattr(df.set_index('date').index, freq)
        grouper = df.groupby('date')
    else:
        grouper = df.set_index('date').resample(freq)

    if col:
        grouper[col].mean().plot.bar()
    else:
        grouper.size().plot.bar()

def plot_grouped_ts_bar(df, cat, col=None, freq='M'):
    df = df.copy()

    if freq in ['month', 'dow', 'hour']:
        df['date'] = getattr(df.set_index('date').index, freq)
    else:
        df = df.set_index('date').to_period(freq).reset_index()

    if col:
        df.pipe(crosstab, 'date', cat, col).plot.bar()
    else:
        df.pipe(crosstab, 'date', cat).plot.bar()

# needs column named date
# need to adjust ylim
def plot_ts_box(df, col, freq='M'):
    df = df.copy()

    if freq in ['month', 'dow', 'hour']:
        df['date'] = getattr(df.set_index('date').index, freq)
    else:
        df = df.set_index('date').to_period(freq)

    df.boxplot(by='date', column=col)

    plt.xticks(rotation=90)

def plot_correlation_matrix(X, **kwargs):
    sns.heatmap(X.corr(), annot=True, fmt='.2f', **kwargs)

def plot_confusion_matrix(model, X, y, threshold=0.5, prop=False):
    try:
        c = confusion_matrix(y, model.predict_proba(X)[:, 1] > threshold)
    except:
        c = confusion_matrix(y, model.predict(X))

    if prop:
        c = c / float(sum(sum(c)))
        sns.heatmap(c, annot=True, fmt='.2f')
    else:
        sns.heatmap(c, annot=True, fmt='d')

def plot_learning_curves(model, X, y):
    sizes, t_scores, v_scores = learning_curve(model, X, y, cv=10, scoring='roc_auc')
    plt.plot(sizes, np.mean(t_scores, axis=1), label='train')
    plt.plot(sizes, np.mean(v_scores, axis=1), label='test')
    plt.legend(loc=(1, 0.5))

# reduce number of arguments?
def plot_roc_curves(model, X_train, y_train, X_test, y_test):

    fpr1, tpr1, _ = roc_curve(y_train, model.predict_proba(X_train)[:, 1])
    plt.plot(fpr1, tpr1, color='g', label='train')

    fpr2, tpr2, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.plot(fpr2, tpr2, color='b', label='test')

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.legend(loc=(1, 0.5))
