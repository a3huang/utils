import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

def plot_missing(df, n=5):
    a = df.isnull().mean(axis=0)
    a = a[a > 0]

    if len(a) == 0:
        return 'No Missing Values'

    b = a.sort_values(ascending=False)[:n]
    b[::-1].plot.barh()

    plt.xlabel('proportions')

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

def plot_grouped_bar1(df, cat, col, is_cat=True, **kwargs):
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

    plt.title('%s vs. %s' % (cat, col))

# col should be continuous or binary
def plot_grouped_bar2(df, cat1, cat2, col, stacked=False, **kwargs):
    df = df.copy()

    cat1 = _index_to_name(df, cat1)
    cat2 = _index_to_name(df, cat2)

    df[cat1] = _top_n_cat(df[cat1])
    df[cat2] = _top_n_cat(df[cat2])

    fig, ax = plt.subplots()

    if stacked:
        pd.crosstab(df[cat1], df[cat2], df[col], aggfunc=np.mean)\
            .sort_index(axis=0, ascending=False)\
            .plot.barh(ax=ax, stacked=True,
                       color=reversed(plt.rcParams['axes.color_cycle'][:5]), **kwargs)
        ax.legend(title=cat2, loc=(1, 0.5))

    else:
        pd.crosstab(df[cat1], df[cat2], df[col], aggfunc=np.mean)\
            .sort_index(axis=0, ascending=False)\
            .sort_index(axis=1, ascending=False)\
            .plot.barh(ax=ax, **kwargs)

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
def plot_grouped_means2(df, cat1, cat2, col):
    df = df.copy()

    cat1 = _index_to_name(df, cat1)
    cat2 = _index_to_name(df, cat2)
    col = _index_to_name(df, col)

    df[cat1] = _top_n_cat(df[cat1])
    df[cat2] = _top_n_cat(df[cat2])

    pd.crosstab(df[cat1], df[cat2], df[col], aggfunc=np.mean).plot()

    plt.legend(title=cat2, loc=(1, 0.5))
    plt.ylabel(col)
    plt.title('%s vs. %s' % (cat1, cat2))

def plot_heatmap(df, cat1, cat2, col):
    df = df.copy()

    cat1 = _index_to_name(df, cat1)
    cat2 = _index_to_name(df, cat2)
    col = _index_to_name(df, col)

    df[cat1] = _top_n_cat(df[cat1])
    df[cat2] = _top_n_cat(df[cat2])

    sns.heatmap(pd.crosstab(df[cat1], df[cat2], df[col], aggfunc=np.mean))

    plt.title(col)

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

def plot_faceted_hist(df, cat, col, **kwargs):
    df = df.copy()

    cat = _index_to_name(df, cat)
    col = _index_to_name(df, col)

    df[cat] = _top_n_cat(df[cat])

    col_order = df[cat].value_counts().sort_index().index

    g = sns.FacetGrid(df, col=cat, col_wrap=4, col_order=col_order)
    g.map(plt.hist, col, **kwargs)
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

    g = sns.FacetGrid(df, col=cat, col_wrap=4)
    g.map(sns.distplot, col, hist=False, **kwargs)
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

    g = sns.FacetGrid(df, col=cat1, col_wrap=4)
    g.map(sns.boxplot, cat2, col, **kwargs)

def plot_grouped_scatter(df, cat, cont1, cont2, ax=None):
    df = df.copy()

    cat = _index_to_name(df, cat)
    cont1 = _index_to_name(df, cont1)
    cont2 = _index_to_name(df, cont2)

    df[cat] = _top_n_cat(df[cat])
    top = df[cat].value_counts()
    df[cat] = df[cat].map(dict(zip(top.index, range(6))))

    grouped = df.groupby(cat)

    colors = plt.rcParams['axes.color_cycle'][:5]

    if ax is None:
        fig, ax = plt.subplots()

    for key, group in grouped:
        group.plot.scatter(cont1, cont2, label=key, color=colors[int(key)], ax=ax)

    plt.legend(title=cat, loc=(1, 0.5))
    plt.title('%s vs. %s' % (cont1, cont2))

def plot_pca_components(df, cat, model=None, n=1000, ax=None):
    df = df.copy()
    s = StandardScaler()

    if model is None:
        model = PCA()

    if n is not None:
        df = df.sample(n)

    X = df[df.columns.difference([cat])]

    df['Component 1'] = model.fit_transform(s.fit_transform(X))[:, 0]
    df['Component 2'] = model.fit_transform(s.fit_transform(X))[:, 1]

    plot_grouped_scatter(df, cat, 'Component 1', 'Component 2', ax=ax)
