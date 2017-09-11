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

from utils.data import table

#####
def barplot(df, col, by=None, val=None, prop=False, return_obj=False):
    # how to adjust order of bars
    '''
    Plot a bar plot or a grouped bar plot for a categorical column.

    ex) df.pipe(barplot, by=cat, col=col, prop=True)
    '''

    df = df.copy()

    if by and val:
        #a = df.pipe(table, by, col, val)
        a = df.pipe(table, by, col, val).stack().reset_index()
        sns.barplot(y=by, x=0, hue=col, orient='h', data=a)
    elif by:
        df['count'] = 1. / (df.groupby(by)[col].transform('size') if prop else 1)
        sns.barplot(y=by, x='count', hue=col, orient='h', data=df, estimator=np.sum)
        # a = df.pipe(table, by, col)
        # a = a.div(a.sum(axis=1) if prop else 1, axis=0)
    elif not (by or val):
        df['count'] = 1. / (len(df) if prop else 1)
        sns.barplot(y=col, x='count', orient='h', data=df, estimator=np.sum)
        #a = df[col].value_counts()
        #a = a.div(a.sum() if prop else 1)
    else:
        raise Exception, "Invalid combination of arguments."

    # if return_obj:
    #     return a
    # else:
    #     a.plot.barh()
    #     plt.gca().invert_yaxis()
    #     plt.legend(title=col, loc=(1, 0))

def boxplot(df, col, by, hue=None, orient='h'):
    '''
    Plot a grouped box plot for a continuous column.

    ex) df.pipe(boxplot, by=cat, col=col)
    '''

    order = df.groupby(by)[col].median().sort_values().index

    if orient == 'h':
        x, y = col, by
    elif orient == 'v':
        x, y = by, col

    if hue:
        sns.boxplot(x, y, hue=hue, data=df, orient=orient, order=order)
    else:
        sns.boxplot(x, y, data=df, orient=orient, order=order)

def heatmap(df, col=None, by=None, val=None, **kwargs):
    '''
    Plot a heatmap to compare two categorical columns.
    Plot an interaction heatmap between two predictor columns and a target column.
    Plot a heatmap of a table of values.

    ex) df.pipe(heatmap, col, by)
    ex) df.iloc[:, 4:10].corr().pipe(heat)
    '''

    if col and by and val:
        sns.heatmap(df.pipe(table, col, by, val), annot=True, fmt='.2f', **kwargs)
    elif col and by:
        sns.heatmap(df.pipe(table, col, by), annot=True, fmt='.2f', **kwargs)
    elif not (col or by or val):
        sns.heatmap(df, annot=True, fmt='.2f', **kwargs)
    else:
        raise Exception, "Invalid combination of arguments."

def histogram(df, col, by=None, range=None, prop=False):
    '''
    Plot a histogram or a grouped histogram for a continuous column.

    ex) df.pipe(histogram, col, prop=True)
    '''

    df[col].hist(range=range)
    ticks = plt.xticks()[0]
    range = ticks[1], ticks[-2]
    total_length = range[1] - range[0]
    bin_size = ticks[1] - ticks[0]
    bins = int(total_length / bin_size)
    plt.clf()

    if by:
        for i, a in df.groupby(by)[col]:
            weights = np.ones_like(a) / float(len(a) if prop else 1)
            a.plot.hist(range=range, bins=bins, weights=weights, alpha=0.4,
                        label=i)
        plt.legend(title=by, loc=(1, 0))

    else:
        weights = np.ones_like(df[col]) / float(len(df[col]) if prop else 1)
        df[col].plot.hist(range=range, bins=bins, weights=weights, alpha=0.4)

def distplot(df, col, by=None):
    '''
    Plot a density plot or a grouped density plot for a continuous column.

    ex) df.pipe(distplot, col)
    '''

    if by:
        df.groupby(by)[col].plot(kind='density')
        plt.legend(title=by, loc=(1, 0))

    else:
        df[col].plot(kind='density')

def lineplot(df, col, by, val):
    '''
    Plot an interaction lineplot between two predictor columns and a target column.

    ex) df.pipe(lineplot, col1, col2, target)
    '''

    df.pipe(table, col, by, val).plot()
    plt.legend(title=by, loc=(1, 0))

def scatterplot(df, x, y, by=None):
    '''
    Plot a scatter plot for 2 continuous variables. Group by an optional 3rd variable
    using color.

    ex) df.pipe(scatterplot, x, y, by=target)
    '''

    if by:
        sns.lmplot(x, y, hue=by, ci=False, data=df)
        plt.legend(loc=(1, 0))

    else:
        sns.lmplot(x, y, data=df, ci=False)

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

def facet(df, row, col, **kwargs):
    '''
    Convenience function to create facet grid.

    ex) df.pipe(facet, 'Generation', 'Legendary').map(plt.scatter, 'Attack', 'Defense')
    '''
    return sns.FacetGrid(df, row=row, col=col, **kwargs)
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
