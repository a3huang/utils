import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_missing(df, n=5):
    a = df.isnull().mean(axis=0)
    a = a[a > 0]

    if len(a) == 0:
        return 'No Missing Values'

    b = a.sort_values(ascending=False)[:n]
    b[::-1].plot.barh()

    df.xlabel('proportions')

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

def plot_grouped_bar(df, col1, col2, stacked=False):
    df = df.copy()

    col1 = _index_to_name(df, col1)
    col2 = _index_to_name(df, col2)

    df[col1] = _top_n_cat(df[col1])
    df[col2] = _top_n_cat(df[col2])

    df.groupby(col1)[col2].mean().sort_index(ascending=False).plot.barh(stacked=stacked)

    plt.xlabel('proportions')
    plt.title('%s vs. %s' % (col1, col2))
