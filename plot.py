import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def _index_to_name(df, col):
    if isinstance(col, int):
        col = df.columns[col]
    return col

def _top_n_cat(a, n=5):
    counts = a.value_counts(dropna=False)
    top = counts.iloc[:n].index
    return a.apply(lambda x: x if x in top else 'other')

def plot_missing(df, n=5):
    a = df.isnull().mean(axis=0)
    a = a[a > 0]

    if len(a) == 0:
        return 'No Missing Values'

    b = a.sort_values(ascending=False)[:n]
    b[::-1].plot.barh()

    df.xlabel('proportions')

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

# col should be contiuous or binary
def plot_grouped_bar(df, cat, col, stacked=False):
    df = df.copy()

    cat = _index_to_name(df, cat)
    col = _index_to_name(df, col)

    df[cat] = _top_n_cat(df[cat])

    df.groupby(cat)[col].mean().sort_index(ascending=False).plot.barh(stacked=stacked)

    plt.xlabel('proportions')
    plt.title('%s vs. %s' % (cat, col))

# col should be continuous or binary
def plot_grouped_bar2(df, cat1, cat2, col, stacked=False):
    df = df.copy()

    cat1 = _index_to_name(df, cat1)
    cat2 = _index_to_name(df, cat2)

    df[cat1] = _top_n_cat(df[cat1])
    df[cat2] = _top_n_cat(df[cat2])

    fig, ax = plt.subplots()

    if stacked:
        pd.crosstab(df[cat1], df[cat2], df[col], aggfunc=np.mean)\
            .sort_index(axis=0, ascending=False)\
            .plot.barh(ax=ax, stacked=True, color=reversed(plt.rcParams['axes.color_cycle'][:5]))

    else:
        pd.crosstab(df[cat1], df[cat2], df[col], aggfunc=np.mean)\
            .sort_index(axis=0, ascending=False)\
            .sort_index(axis=1, ascending=False)\
            .plot.barh(ax=ax, legend='reverse')

    plt.legend(title=cat2, loc=(1, 0.5))
    plt.xlabel(col)
    plt.title('%s vs. %s' % (cat1, cat2))
