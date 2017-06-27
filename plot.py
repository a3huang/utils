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

from model import _get_feature_importances
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
def winsorize(x, p=0.5):
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

def plot_heatmap_groupby_2(df, cat1, cat2, col, top=20):
    df = df.copy()

    df[cat1] = _top_n_cat(df[cat1], top)
    df[cat2] = _top_n_cat(df[cat2], top)

    a = pd.crosstab(df[cat1], df[cat2], df[col], aggfunc=np.mean)
    sns.heatmap(a)
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

def plot_hist_single_column(arg1, arg2=None, winsorize_column=True, **kwargs):
    if arg2 is None:
        assert len(arg1.shape) == 1, 'If only one argument, then must be single column'
        a = arg1
        col = a.name
    else:
        df = arg1
        col = arg2
        a = df[col]

    if winsorize_column:
        a = winsorize(a)

    assert len(a.shape) == 1, 'Must be single column'

    weights = np.ones_like(a) / float(len(a))
    a.plot.hist(weights=weights, **kwargs)
    plt.ylabel('Proportion')
    plt.title(col)

def plot_hist_groupby_1(df, cat, col, bins=40, top=20, winsorize_column=True, facet=True,
                        col_wrap=4, alpha=0.3, **kwargs):
    df = df.copy()

    df[cat] = _top_n_cat(df[cat], top)

    if facet == True:
        g = sns.FacetGrid(df, col=cat, col_wrap=col_wrap, **kwargs)
        g.map(plot_hist_single_column, col, winsorize_column=winsorize_column, bins=bins)
        return g

    if winsorize_column:
        df[col] = winsorize(df[col])
        df = df[~df[col].isnull()]

    assert df[col].isnull().any() == False, 'Column contains null values'

    bins = np.histogram(df[col], bins=bins)[1]
    groups = df.groupby(cat)[col]
    for k, v in groups:
        plot_hist_single_column(v, bins=bins, label=str(k), alpha=0.3, **kwargs)

    plt.legend(title=cat, loc=(1, 0.5))
    plt.title(col)

def plot_density_groupby_1(df, cat, col, winsorize_column=True, **kwargs):
    df = df.copy()

    df[cat] = _top_n_cat(df[cat])

    if winsorize_column:
        df[col] = winsorize(df[col])
        df = df[~df[col].isnull()]

    assert df[col].isnull().any() == False, 'Column contains null values'

    groups = df.groupby(cat)[col]
    for k, v in groups:
        sns.kdeplot(v, shade=True, label=str(k), **kwargs)

    plt.legend(title=cat, loc=(1, 0.5))
    plt.title(col)




def plot_faceted_density(df, cat, col, **kwargs):
    df = df.copy()

    cat = _index_to_name(df, cat)
    col = _index_to_name(df, col)

    df[cat] = _top_n_cat(df[cat])

    g = sns.FacetGrid(df, col=cat, **kwargs)
    g.map(sns.kdeplot, col, shade=True)
    g.set_xticklabels(rotation=45)

# seems to have errors with astype(ordered=True)
def plot_grouped_box(df, col1, col2, **kwargs):
    df = df.copy()

    col1 = _index_to_name(df, col1)
    col2 = _index_to_name(df, col2)

    df[col1] = _top_n_cat(df[col1])

    df[col1] = df[col1].astype('category', ordered=True)
    a = df[col1].cat.categories
    df[col1] = df[col1].cat.reorder_categories(list(reversed(a)))

    df.boxplot(by=col1, column=col2, vert=False, **kwargs)
    plt.xlabel(col2)
    plt.ylabel(col1)
    plt.suptitle('')
    plt.title('%s vs. %s' % (col1, col2))

def plot_faceted_box(df, cat1, cat2, col, **kwargs):
    df = df.copy()

    cat1 = _index_to_name(df, cat1)
    cat2 = _index_to_name(df, cat2)
    col = _index_to_name(df, col)

    df[cat1] = _top_n_cat(df[cat1])
    df[cat2] = _top_n_cat(df[cat2])

    g = sns.FacetGrid(df, col=cat1, **kwargs)
    g.map(sns.boxplot, cat2, col, vert=False).set_xlabels(col).set_ylabels(cat2)

def plot_scatter(df, cont1, cont2, **kwargs):
    sns.lmplot(cont1, cont2, data=df, **kwargs)

# need both ax argument and kwargs here?
def plot_grouped_scatter(df, cat, cont1, cont2, ax=None, **kwargs):
    df = df.copy()

    cat = _index_to_name(df, cat)
    cont1 = _index_to_name(df, cont1)
    cont2 = _index_to_name(df, cont2)

    df[cat] = _top_n_cat(df[cat])
    top = df[cat].value_counts()
    df[cat] = df[cat].map(dict(zip(top.index, range(5))))

    grouped = df.groupby(cat)

    colors = plt.rcParams['axes.color_cycle'][:5]

    if ax is None:
        fig, ax = plt.subplots()

    for key, group in grouped:
        ax.scatter(group[cont1], group[cont2], label=key, color=colors[int(key)])
        ax.set_title('')

    plt.legend(title=cat, loc=(1, 0.5))
    plt.suptitle('%s vs. %s' % (cont1, cont2), y=1.01)

# how to refactor scatter_plot function?
def plot_faceted_scatter(df, cat1, cat2, cont1, cont2, **kwargs):

    def scatter_plot(x, y, color=None):
        sns.regplot(x, y, scatter_kws={'color': color})

    df = df.copy()

    cat1 = _index_to_name(df, cat1)
    cat2= _index_to_name(df, cat2)
    cont1 = _index_to_name(df, cont1)
    cont2 = _index_to_name(df, cont2)

    df[cat1] = _top_n_cat(df[cat1])
    df[cat2] = _top_n_cat(df[cat2])

    g = sns.FacetGrid(df, col=cat1, hue=cat2, **kwargs)
    g.map(scatter_plot, cont1, cont2).add_legend()

def plot_pca(df, cat, model=None, n=1000, **kwargs):
    df = df.copy()
    s = StandardScaler()

    if model is None:
        model = PCA()

    if n is not None:
        df = df.sample(n)

    X = df[df.columns.difference([cat])]

    df['Component 1'] = model.fit_transform(s.fit_transform(X))[:, 0]
    df['Component 2'] = model.fit_transform(s.fit_transform(X))[:, 1]

    plot_grouped_scatter(df, cat, 'Component 1', 'Component 2', **kwargs)

def plot_faceted_pca(df, cat, models, **kwargs):
    r = int(len(models) / 2.0)

    fig, ax = plt.subplots(r, 2, **kwargs)
    ax = ax.flatten()

    for i, j in zip(models, ax):
        df.pipe(plot_pca, cat, model=i, ax=j)
        j.set_title(repr(i.__class__).split('.')[-1].split("'")[0][:15])

        if j != ax[-1]:
            j.legend().remove()

    plt.tight_layout()

# need to allow plotting more than 5 cluster colors
def plot_clusters(df, model=None, pca_model=None, n=1000, **kwargs):
    df = df.copy()
    s = StandardScaler()

    if model is None:
        model = KMeans(n_clusters=5)

    if n is not None:
        df = df.sample(n)

    model.fit(s.fit_transform(df))

    df['cluster'] = model.labels_

    fig, ax = plt.subplots(figsize=figsize)
    plot_pca(df, 'cluster', pca_model, n=None, **kwargs)

# need to fix legend handling
def plot_faceted_clusters(df, cat, cluster_col, **kwargs):
    clusters = df[cluster_col].unique()
    r = len(clusters) / 2

    fig, ax = plt.subplots(r, 2, **kwargs)
    ax = ax.flatten()

    for i, j in zip(clusters, ax):
        df[df[cluster_col] == i].pipe(plot_pca, cat, n=None, ax=j)

    plt.tight_layout()

# combine model and X into one object?
def plot_feature_importances(model, X, **kwargs):
    a = pd.DataFrame(sorted(zip(X.columns, _get_feature_importances(model)),
                        key=lambda x: abs(x[1]), reverse=True))

    a.sort_index(ascending=False).set_index(0).plot.barh(**kwargs)
    plt.legend().remove()
    plt.ylabel('Feature')
    plt.title('Feature Importance Measure')

# integrate this better with barplot function above?
def plot_top_word_frequencies(documents, prop=True, **kwargs):
    c = CountVectorizer(**kwargs)
    c.fit(documents)

    counts = c.fit_transform(documents).sum(axis=0)
    vocab = pd.DataFrame([(k, counts[0, c.vocabulary_[k]]) for k, v in c.vocabulary_.items()][:10])
    total = vocab[1].sum()
    vocab['freq'] = vocab[1] / total

    if prop == True:
        vocab[[0, 'freq']].set_index(0).sort_values(by='freq').plot.barh(**kwargs)
    else:
        vocab[[0, 1]].set_index(0).sort_values(by=1).plot.barh(**kwargs)

    plt.ylabel('words')
    plt.legend().remove()

def draw_tree(X, y, filename, **kwargs):
    model = tree.DecisionTreeClassifier(**kwargs)
    model.fit(X, y)
    dot_data = tree.export_graphviz(model,
                                    out_file=None,
                                    feature_names=X.columns,
                                    filled=True,
                                    class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(filename)

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
