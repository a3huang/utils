import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, roc_curve, f1_score
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from rpy2.robjects import pandas2ri, r
from sklearn import tree
import itertools
import pydotplus
import subprocess

from utils.data import table, rename

import os

#
def dummy_categorical(df, n):
    '''
    Create a categorical column with labels 1 to n for testing purposes.

    ex) df['dummy'] = df.pipe(dummy_categorical, 5)
    '''

    size = df.shape[0] / n
    remainder = df.shape[0] - n * size

    a = []
    a.append(np.full((size + remainder, 1), 1))

    for i in range(2, n+1):
        a.append(np.full((size, 1), i))

    return pd.DataFrame(np.append(a[0], a[1:]))

#
def dummy_continuous(df, loc=0, scale=1):
    '''
    Create a continuous column from a normal distribution for testing purposes.

    ex) df['dummy'] = df.pipe(dummy_continuous)
    '''

    return pd.DataFrame(np.random.normal(loc=loc, scale=scale, size=df.shape[0]))

#
def facet(df, row, col, **kwargs):
    '''
    Convenience function for creating seaborn facet grids.

    ex) df.pipe(facet, row='Type 1', col='Type 2').map(plt.scatter, 'Attack', 'Defense')
    '''

    return sns.FacetGrid(df, row=row, col=col, **kwargs)

#
def nice_range_bin(ax, range=None):
    '''
    Helper function for matplotlib histograms to find the right range and number
    of bins so that the bar edges will line up nicely with the x-axis tick marks.

    ex) fig, ax = plt.subplots()
        ax.hist(a)
        range, bins = nice_range_bin(ax)
    '''

    ticks = ax.get_xticks()

    if range is None:
        range = ticks[1], ticks[-2]

    total_length = range[1] - range[0]
    bin_size = ticks[1] - ticks[0]
    bins = int(total_length / bin_size)
    return range, bins

#
def ceil_with_base(x, base):
    '''
    Take the ceiling of a number with respect to any base.

    ex) ceil_with_base(0.03, 0.05) -> 0.05
    ex) ceil_with_base(0.02, 0.05) -> 0.05
    '''

    return base * np.ceil(float(x) / base)

#
def round_with_base(x, base):
    '''
    Round a number with respect to any base.

    ex) round_with_base(0.03, 0.05) -> 0.05
    ex) round_with_base(0.02, 0.05) -> 0
    '''

    return base * np.round(float(x) / base)

#
def nice_round(x):
    '''
    Round a number "nicely" to the nearest denomination of 5 or 10. Numbers less
    than 1 will be always be rounded upwards. Numbers greater than 1 may be rounded
    upwards or downwards.

    ex) nice_round(0.03) -> 0.05
    ex) nice_round(0.02) -> 0.05
    ex) nice_round(12) -> 10
    ex) nice_round(14) -> 15
    ex) nice_round(7) -> 7
    ex) nice_round(7.2) -> 7
    '''

    power = np.ceil(np.abs(np.log10(x))) - 1

    if x < 1:
        return ceil_with_base(x, base=5*10**-(power+1))

    else:
        return round_with_base(x, base=5*10**(power-1))

#
def nice_hist(a):
    '''
    Creates a "nice" histogram by adjusting both the bin edges and the x-axis tick
    marks so that they line up nicely.

    '''

    _, edges = np.histogram(a)
    width = nice_round(edges[1] - edges[0])
    new_edges = np.unique([round_with_base(i, width) for i in edges])
    plt.hist(a, range=(new_edges[0], new_edges[-1]), bins=len(new_edges)-1)
    plt.xticks(new_edges)

#
def barplot(df, col, by=None, kind=None, prop=False):
    '''
    Create a bar plot for a categorical variable. Group by an optional 2nd
    categorical variable for a grouped bar plot.

    ex) df.pipe(bar, col='HP', by='Type')
    '''

    if by:
        if prop:
            normalize = 'index'
        else:
            normalize = False

        data = df.pipe(table, col, by, normalize=normalize).unstack().reset_index()

        if kind == 'facet':
            g = sns.factorplot(x=0, y=col, col=by, data=data, kind='bar', orient='h')
            g.set_axis_labels('', col)

        elif kind == 'stack':
            values = data[by].unique()
            colors = sns.color_palette()
            for i in reversed(range(1, len(values)+1)):
                layer = data[data[by].isin(values[:i])]
                sns.barplot(x=0, y=col, data=layer, estimator=np.sum, ci=False,
                            orient='h', color=colors[i-1])

        else:
            sns.barplot(x=0, y=col, hue=by, data=data, orient='h')

    else:
        data = df[col].value_counts(normalize=prop).reset_index()
        sns.barplot(x=col, y='index', data=data, orient='h')
        plt.ylabel(col)

    plt.xlabel('')
    plt.legend(title=by, loc=(1, 0))

def boxplot(df, col, by, sort_median=False):
    '''
    Create a grouped box plot for a continuous variable.

    ex) df.pipe(box, col='HP', by='Type')
    '''

    if sort_median:
        order = df.groupby(by)[col].median().sort_values().index
    else:
        order = None

    sns.boxplot(x=col, y=by, data=df, order=order, orient='h')

def distplot(df, col, by=None, prop=False, range=None, facet=False):
    '''
    Create a histogram for a continuous variable. Group by an optional 2nd
    categorical variable and for a grouped density plot.

    ex) df.pipe(hist, col='HP', by='Type')
    '''

    if by:
        if facet:
            g = sns.FacetGrid(df, col=by)
            g.map(plt.hist, col, range=range)
            range, bins = nice_hist_params(g.axes.flat[0], range)
            plt.clf()
            g = sns.FacetGrid(df, col=by, col_wrap=3)
            g.map(plt.hist, col, range=range, bins=bins)
        else:
            df.groupby(by)[col].plot(kind='density')
            plt.xlim(range)
            plt.legend(title=by, loc=(1, 0))

    else:
        if prop:
            weights = np.ones_like(df[col]) / float(len(df[col]))
        else:
            weights = None

        fig, ax = plt.subplots()
        df[col].hist(range=range, ax=ax)
        range, bins = nice_range_bin(ax, range)
        plt.clf()

        df[col].plot.hist(range=range, bins=bins, weights=weights, alpha=0.4)

def heatplot(df, x, y, z=None, normalize=False):
    '''
    Create a heatmap between 2 categorical variables. Calculate the mean for an
    optional 3rd continuous variable.

    ex) df.pipe(heat, x='Type 1', y='Type 2', z='Attack')
    '''

    if z:
        sns.heatmap(df.pipe(table, x, y, z), annot=True, fmt='.2f')
    else:
        sns.heatmap(df.pipe(table, x, y, normalize=normalize), annot=True, fmt='.2f')

def scatplot(df, x, y, by=None):
    '''
    Create a scatter plot for 2 continuous variables. Group by an optional 3rd
    categorical variable.

    ex) df.pipe(scat, x='Attack', y='Defense', by='Generation')
    '''

    sns.lmplot(x=x, y=y, hue=by, data=df, legend=False, fit_reg=False, ci=False)

    if by:
        plt.legend(title=by, loc=(1, 0))

def interactplot(df, col, by, val, heat=False):
    '''
    Create an interaction lineplot or heatmap between 2 predictor variables and
    a 3rd target variable.

    ex) df.pipe(interactions, col='Type 1', by='Type 2', val='Attack')
    '''

    a = df.pipe(table, col, by, val)

    if heat:
        sns.heatmap(a, annot=True, fmt='.2f')
    else:
        a.plot()
        plt.legend(title=by, loc=(1, 0))
###

def tsplot(df, date, by=None, val=None, freq='M', kind='line'):
    '''
    Plot a time series.

    ex) df.pipe(tsplot, date, by=col)
    '''

    if kind not in ['area', 'bar', 'line']:
        raise Exception, 'Invalid plot type'

    if by and val:
        df.set_index('date').groupby(by).resample(freq)[val].mean().unstack(by).plot(kind=kind)
    elif by:
        df.set_index('date').groupby(by).resample(freq).size().unstack(by).plot(kind=kind)
    elif val:
        df.set_index('date').resample(freq)[val].mean().plot(kind=kind)
    else:
        df.set_index('date').resample(freq).size().plot(kind=kind)

def tsboxplot(df, date, col, freq='M'):
    '''
    Plot a boxplot of a continuous column for each aggregated time window.

    ex) df.pipe(tsboxplot, date, col=col)
    '''

    sns.boxplot(x=date, y=col, data=df.set_index(date).to_period(freq).reset_index())

def facet_histogram(df, row, val, col=None):
    # how to add prop to histograms?
    '''
    Convenience function to facet either by 2 categorical variables or 2
    continuous variables.

    ex) df.pipe(facet_histogram, cat, [col1, col2])
    '''

    df.pipe(facet, row, col).map(plt.hist, val)

def generate_boxplots(df, by, folder_name, default_dir='/Users/alexhuang/'):
    directory = default_dir + folder_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i, col in enumerate(df.columns.difference([by])):
        sns.boxplot(df[by], df[col])
        plt.savefig(directory + '%s.png' % i)
        plt.close()
#####

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
    plot_scatter(df, cat, 'PCA 1', 'PCA 2', fit_reg=False, **kwargs)
    return df

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
    return df

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

def plot_feature_importances(model, X, top=None, **kwargs):
    #X = df[df.columns.difference([target])]
    a = _get_feature_importances(model, X)
    model_name = _get_model_name(model)

    if top:
        a = a[:top]

    a.sort_index(ascending=False).set_index(0).plot.barh(**kwargs)
    plt.ylabel('Feature')
    plt.title(model_name)
    plt.legend().remove()
    return a

def plot_feature_scores(df, scores, top=5):
    pd.DataFrame(sorted(zip(df.columns, scores), key=lambda x: x[1], reverse=True)[:top])\
        .set_index(0).sort_values(by=1).plot.barh()

def plot_confusion_matrix(model, X, y, threshold=0.5, norm_axis=1, **kwargs):
    # X = df.drop(target, 1)
    # y = df[target]

    try:
        prediction = model.predict_proba(X)[:, 1] > threshold
    except:
        prediction = model.predict(X)

    a = confusion_matrix(y, prediction).astype(float)
    a = np.divide(a, np.sum(a, norm_axis))
    sns.heatmap(a, annot=True, fmt='.2f', **kwargs)
    plt.ylabel('True')
    plt.title('Predicted')
    return a

def plot_roc_curve(model, X, y):
    # X = df.drop(target, 1)
    # y = df[target]

    #model_name = _get_model_name(model)

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

def plot_learning_curves(model, X, y):
    # X = df.drop(target, 1)
    # y = df[target]

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

def plot_probs(model, X_test, **kwargs):
    plt.hist(model.predict_proba(X_test)[:, 1], **kwargs)

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

def loess(df, col1, col2):
    pandas2ri.activate()
    dfr = pandas2ri.py2ri(df)

    a = r.loess('%s ~ %s' % (col2.replace('/', '.'), col1.replace('/', '.')), data=dfr)

    x = pd.DataFrame(np.array(a.rx2('x')))
    y = pd.DataFrame(np.array(a.rx2('fitted')))

    x_sorted = x.sort_values(by=0).index

    return x.loc[x_sorted], y.loc[x_sorted]

def compare_feature_sets_boot(a, b, col):
    a['data'] = 1
    b['data'] = 2

    c = pd.concat([a, b])[[col, 'data']]
    c.columns = ['Score_%s' % i for i in range(1, len(c.columns))] + ['data']

    sns.boxplot(y='variable', x='value', hue='data', data=c.melt('data'), orient='h')
    plt.legend(loc=(1,.5))

def compare_feature_sets_bar(a, b, col):
    a['data'] = 1
    b['data'] = 2

    pd.concat([a, b], axis=1)[[col]].plot.bar()
    plt.legend(loc=(1,.5))

def plot_f1(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True)

    l = []
    for i in np.linspace(.1,1,10):
        l1 = []
        for tr, te in cv.split(X, y):
            true = y.values[te]
            pred = model.predict_proba(X.values[te])[:, 1] > i
            l1.append(f1_score(true, pred))
        l.append(np.mean(l1))

    plt.plot(np.linspace(.1,1,10), l)

def plot_true_pred(model, X_test, y_test):
    xlims = model.predict(X_test).min(), model.predict(X_test).max()
    ylims = y_test.min(), y_test.max()
    range_min = min(xlims[0], ylims[0])
    range_max = max(xlims[1], ylims[1])
    plt.scatter(y_test, model.predict(X_test))
    plt.plot([range_min, range_max], [range_min, range_max], linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
