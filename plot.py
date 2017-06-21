import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

from sklearn import tree
import pydotplus

from model import _get_feature_importances
from data import crosstab

def _index_to_name(df, col):
    if isinstance(col, int):
        col = df.columns[col]
    return col

def _top_n_cat(a, n=5):
    counts = a.value_counts(dropna=False)
    top = counts.iloc[:n].index
    return a.apply(lambda x: x if x in top else 'other')

def winsorize(df, col, p):
    n = int(1/p)
    sorted_col = df[col].sort_values().reset_index(drop=True)
    quantiles = pd.qcut(sorted_col.reset_index()['index'], n).cat.codes
    a = pd.concat([sorted_col, quantiles], axis=1)
    quantiles_to_keep = a[0].unique()[1:-1]
    return a[a[0].isin(quantiles_to_keep)][col]

################################################################################

def plot_missing(df, n=5, **kwargs):
    a = df.isnull().mean(axis=0)
    a = a[a > 0]

    if len(a) == 0:
        return 'No Missing Values'

    b = a.sort_values(ascending=False)[:n]
    b[::-1].plot.barh(ax=ax, **kwargs)

    ax.set_xlabel('proportions')

# col should be categorical
def plot_bar(df, col=None, prop=True, **kwargs):
    if col:
        col = _index_to_name(df, col)
        a = df[col]
    else:
        col = df.name
        a = df.copy()

    a = _top_n_cat(a)
    a = a.value_counts(dropna=False)

    if prop == True:
        a = a / float(sum(a))
        xlabel = 'proportions'
    else:
        xlabel = 'counts'

    a.sort_index(ascending=False).plot.barh(**kwargs)
    plt.xlabel(xlabel)
    plt.title(col)

# how to pick specific class rates to show?
# rename is_cat?
def plot_grouped_bar1(df, cat, col, is_cat=False, **kwargs):
    df = df.copy()

    cat = _index_to_name(df, cat)
    col = _index_to_name(df, col)

    df[cat] = _top_n_cat(df[cat])

    if is_cat or df[col].dtype == 'O':
        df[col] = _top_n_cat(df[col])
        a = df.groupby(cat)[col].value_counts().unstack().sort_index(ascending=False)
        a.plot.barh(**kwargs)
        plt.xlabel('proportions')
    else:
        a = df.groupby(cat)[col].mean().sort_index(ascending=False)
        a.plot.barh(**kwargs)
        plt.xlabel(col)

    plt.legend(title=col, loc=(1, 0.5))

# col should be continuous or binary
def plot_grouped_bar2(df, cat1, cat2, col, stacked=False, **kwargs):
    df = df.copy()

    cat1 = _index_to_name(df, cat1)
    cat2 = _index_to_name(df, cat2)

    df[cat1] = _top_n_cat(df[cat1])
    df[cat2] = _top_n_cat(df[cat2])

    if stacked:
        pd.crosstab(df[cat1], df[cat2], df[col], aggfunc=np.mean)\
            .sort_index(axis=0, ascending=False)\
            .plot.barh(stacked=True, color=reversed(plt.rcParams['axes.color_cycle'][:5]),
                       **kwargs)
        ax.legend(title=cat2, loc=(1, 0.5))

    else:
        pd.crosstab(df[cat1], df[cat2], df[col], aggfunc=np.mean)\
            .sort_index(axis=0, ascending=False)\
            .sort_index(axis=1, ascending=False)\
            .plot.barh(**kwargs)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], title=cat2, loc=(1, 0.5))

    plt.xlabel(col)
    plt.title('%s vs. %s' % (cat1, cat2))

# col should be continuous or binary
def plot_grouped_means1(df, cat, col, **kwargs):
    df = df.copy()

    cat = _index_to_name(df, cat)
    col = _index_to_name(df, col)

    df[cat] = _top_n_cat(df[cat])

    a = df.groupby(cat)[col].mean().plot(**kwargs)
    plt.xlabel(cat)
    plt.title('%s vs. %s' % (cat, col))

# col should be continuous or binary
def plot_grouped_means2(df, cat1, cat2, col, **kwargs):
    df = df.copy()

    cat1 = _index_to_name(df, cat1)
    cat2 = _index_to_name(df, cat2)
    col = _index_to_name(df, col)

    df[cat1] = _top_n_cat(df[cat1])
    df[cat2] = _top_n_cat(df[cat2])

    pd.crosstab(df[cat1], df[cat2], df[col], aggfunc=np.mean).plot(**kwargs)
    plt.legend(title=cat2, loc=(1, 0.5))
    plt.ylabel(col)
    plt.title('%s vs. %s' % (cat1, cat2))

# not to be passed into pipe
def plot_heatmap(df, **kwargs):
    sns.heatmap(df, **kwargs)

def plot_grouped_heatmap(df, cat1, cat2, col, **kwargs):
    df = df.copy()

    cat1 = _index_to_name(df, cat1)
    cat2 = _index_to_name(df, cat2)
    col = _index_to_name(df, col)

    df[cat1] = _top_n_cat(df[cat1])
    df[cat2] = _top_n_cat(df[cat2])

    fig, ax = plt.subplots()
    ax = sns.heatmap(pd.crosstab(df[cat1], df[cat2], df[col], aggfunc=np.mean))
    ax.collections[0].colorbar.set_label(col, rotation=-90, labelpad=15)

def plot_contours(df, cat1, cat2, col):
    df = df.copy()

    cat1 = _index_to_name(df, cat1)
    cat2 = _index_to_name(df, cat2)
    col = _index_to_name(df, col)

    df[cat1] = _top_n_cat(df[cat1])
    df[cat2] = _top_n_cat(df[cat2])

    sns.interactplot(cat1, cat2, col, data=df)

# add winsorize option?
def plot_hist(df, col=None, prop=True, bins=10, **kwargs):
    if col:
        col = _index_to_name(df, col)
        a = df[col]
    else:
        col = df.name
        a = df.copy()

    a = a.dropna()

    if prop == True:
        weights = np.ones_like(a) / float(len(a))
        ylabel = 'proportions'
    else:
        weights = np.ones_like(a)
        ylabel = 'counts'

    a.hist(weights=weights, bins=bins, **kwargs)
    plt.title(col)

def plot_grouped_hist(df, cat, col, prop=True, **kwargs):
    df = df.copy()

    cat = _index_to_name(df, cat)
    col = _index_to_name(df, col)

    df[cat] = _top_n_cat(df[cat])

    groups = df.groupby(cat)[col]

    fig, ax = plt.subplots()

    for k, v in groups:
        if prop == True:
            weights = np.ones_like(v) / float(len(v))
        else:
            weights = np.ones_like(v)

        v.hist(label=str(k), alpha=0.75, weights=weights, ax=ax, **kwargs)

    ax.legend(title=cat, loc=(1, 0.5))
    ax.set_title(col)

# how to give option of not rotating tick labels
def plot_faceted_hist(df, cat, col, **kwargs):
    df = df.copy()

    cat = _index_to_name(df, cat)
    col = _index_to_name(df, col)

    df[cat] = _top_n_cat(df[cat])

    col_order = df[cat].value_counts().sort_index().index

    g = sns.FacetGrid(df, col=cat, col_order=col_order, **kwargs)
    g.map(plt.hist, col)
    g.set_xticklabels(rotation=45)

def plot_grouped_density(df, cat, col, prop=True, **kwargs):
    df = df.copy()

    cat = _index_to_name(df, cat)
    col = _index_to_name(df, col)

    df[cat] = _top_n_cat(df[cat])

    groups = df.groupby(cat)[col]

    fig, ax = plt.subplots()

    for k, v in groups:
        v.plot.density(label=str(k), ax=ax, **kwargs)

    ax.legend(title=cat, loc=(1, 0.5))
    ax.set_title(col)

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
    g.map(sns.boxplot, cat2, col)

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
