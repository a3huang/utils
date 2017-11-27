import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lifelines import KaplanMeierFitter
from lime.lime_tabular import LimeTabularExplainer
from pandas.api.types import is_numeric_dtype, is_string_dtype
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score, \
        precision_score, recall_score, roc_curve
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, export_graphviz

import graphviz
import os
import pydotplus
import subprocess

from utils.data import *

##########################
##### Basic Plotting #####
##########################
def facet(df, col):
    return sns.FacetGrid(col=col, col_wrap=4, data=df, sharex=False)

def barplot(x, y=None, by=None, orient='v', prop=False, stacked=False, **kwargs):
    '''
    Creates a bar plot of counts for a categorical variable or a bar plot of means
    for a continuous variable grouped by a categorical variable.

    ex) barplot(df.cat)
    ex) barplot(df.cat, by=df.cat)
    ex) barplot(df.cat, y=df.cont)
    ex) barplot(df.cat, y=df.cont, by=df.cat)
    ex) df.pipe(facet, 'cat').map(barplot, 'cat')
    '''

    kind = 'barh' if orient == 'h' else 'bar'

    if y is None and by is None:
        df = x.groupby(x).size()
        df = df / df.sum() if prop else df
        df.plot(kind=kind, **kwargs)

    elif y is None:
        df = pd.concat([x, by], axis=1)
        df = df.groupby([x, by]).size()
        df = df / df.groupby(x).sum() if prop else df
        df.unstack().plot(kind=kind, stacked=stacked, **kwargs)
        plt.legend(loc=(1, 0))

    elif by is None:
        df = pd.concat([x, y], axis=1)
        df.groupby(x)[y.name].mean().plot(kind=kind, **kwargs)

    else:
        df = pd.concat([x, y, by], axis=1)
        df = df.groupby([x, by])[y.name].mean()
        df.unstack().plot(kind=kind, **kwargs)
        plt.legend(loc=(1, 0))

def boxplot(df, col, by, hue=None, orient='v', **kwargs):
    '''
    Creates a box plot for a continuous variable grouped by a categorical variable.

    ex) df.pipe(boxplot, col='cont', by='cat')
    ex) df.pipe(boxplot, col='cont', by='cat', hue='cat')
    ex) df.pipe(boxplot, col='cont', by=pd.qcut(df['cont'], 3))
    '''

    color = sns.color_palette()[0]

    if orient == 'h':
        x, y = col, by
    else:
        x, y = by, col

    sns.boxplot(x=x, y=y, hue=hue, data=df, **kwargs)

    if hue is not None:
        plt.legend(loc=(1, 0))

def densityplot(df, col, by=None, range=None):
    '''
    Creates a density plot for a continuous variable.

    ex) df.pipe(densityplot, col='cont')
    ex) df.pipe(densityplot, col='cont', by='cat')
    ex) df.pipe(densityplot, col='cont', by=pd.qcut(df['cont'], 3))
    '''

    if by is None:
        sns.kdeplot(df[col])
        plt.legend().remove()

    else:
        for g, c in df.groupby(by)[col]:
            sns.kdeplot(c, label=g, shade=True)

        plt.xlim(range)
        plt.legend(loc=(1, 0))

def heatmap(df, row, col, val=None, normalize=False, **kwargs):
    '''
    Creates a heat map of counts for 2 categorical variables or a heat map of
    means for a continuous variable grouped by 2 categorical variables.

    ex) df.pipe(heatmap, row='cat', col='cat')
    ex) df.pipe(heatmap, row='cat', col='cat', val='cont')
    ex) df.pipe(heatmap, row='cat', col=pd.qcut(df['cont'], 3))
    '''

    if val is None:
        sns.heatmap(df.pipe(table, row, col, normalize=normalize),
            annot=True, fmt='.2f', **kwargs)
    else:
        sns.heatmap(df.pipe(table, row, col, val), annot=True, fmt='.2f', **kwargs)

def flexible_bin_range(a, bin_num=None, bin_range=None, bin_width=None):
    bin_range = (a.min(), a.max()) if bin_range is None else bin_range

    if bin_width and bin_num:
        raise Exception, 'Must specify only one of bin_num or bin_width'

    elif bin_width:
        bin_num = int(round((bin_range[1] - bin_range[0]) / bin_width))
        bins = [a.min() + bin_width*i for i in np.arange(bin_num+1)]

    elif bin_num:
        bins = bin_num

    else:
        bins = 10

    return bins, bin_range

def nice_bin_range(a, bin_mult=1, bin_range=None):
    # make temporary plot using just min and max to get the "right" x-axis tick marks
    pd.DataFrame([a.min(), a.max()]).plot.hist(range=bin_range)

    ticks = plt.gca().get_xticks()

    if bin_range is None:
        bin_range = ticks[1], ticks[-2]

    total_span = bin_range[1] - bin_range[0]
    bin_width = ticks[1] - ticks[0]
    num_bins = int(total_span / bin_width)

    # remove temporary plot
    plt.clf()

    return bin_mult*num_bins, bin_range

def histogram(df, col, by=None, nice=False, prop=False, **kwargs):
    '''
    Creates a histogram for a continuous variable.

    ex) df.pipe(histogram, col='cont')
    ex) df.pipe(histogram, col='cont', by='cat')
    ex) df.pipe(histogram, col='cont', by=pd.qcut(df['cont'], 3))
    '''

    a = df[col]

    if nice:
        bins, range = nice_bin_range(a, **kwargs)
    else:
        bins, range = flexible_bin_range(a, **kwargs)

    if by is None:
        weights = np.ones_like(a) / float(len(a)) if prop else None
        a.plot.hist(alpha=0.4, bins=bins, range=range, weights=weights)

    else:
        for g, c in df.groupby(by)[col]:
            weights = np.ones_like(c) / float(len(c)) if prop else None
            c.plot.hist(alpha=0.4, bins=bins, label=g, range=range, weights=weights)

        plt.legend(loc=(1, 0))

def scatterplot(df, x, y, by=None):
    '''
    Creates a scatter plot for 2 continuous variables.

    ex) df.pipe(scatterplot, x='cont', y='cont')
    ex) df.pipe(scatterplot, x='cont', y='cont', by='cat')
    ex) df.pipe(scatterplot, x='cont', y='cont', by=pd.qcut(df['cont'], 3))
    '''

    sns.lmplot(x=x, y=y, hue=by, data=df, legend=False)

    if by is not None:
        plt.legend(loc=(1, 0))

################################
##### Time Series Plotting #####
################################
def tsbarplot(df, date, val=None, by=None, unit='weekday'):
    '''
    Creates a bar plot of counts of a time unit (e.g. day of week, hour, minute) for
    a datetime variable.

    ex) df.pipe(tsbarplot, date='datetime')
    ex) df.pipe(tsbarplot, date='datetime', val='cont')
    ex) df.pipe(tsbarplot, date='datetime', val='cont', by='cat')
    ex) df.pipe(tsbarplot, date='datetime', by=pd.qcut(df['cont'], 3))
    '''

    timeunit = getattr(df[date].dt, unit)

    if val is None:
        df.pipe(barplot, col=timeunit, by=by)
    else:
        df.pipe(barplot, col=val, by=timeunit, hue=by)

def tsboxplot(df, date, val, hue=None, freq='M'):
    '''
    Creates a time series box plot for a continuous variable.

    ex) df.pipe(tsboxplot, date='date', val='cont')
    ex) df.pipe(tsboxplot, date='date', val='cont', hue='cat')
    '''

    date = pd.Grouper(key=date, freq=freq)

    columns = []
    for g, c in df.groupby(date):
        a = pd.DataFrame(c[val]).reset_index(drop=True).rename(columns={val: g}).melt()

        if hue is not None:
            b = pd.DataFrame(c[hue]).reset_index(drop=True)
            a = pd.concat([a, b], axis=1)

        columns.append(a)

    data = pd.concat(columns)
    sns.boxplot(x='variable', y='value', hue=hue, data=data)
    plt.xlabel(val)
    plt.xticks(rotation=90)

def tsheatmap(df, date, val=None, freq='M', unit='weekday'):
    '''
    Creates a heat map of counts or a heat map of means for a continuous variable
    grouped by a time unit (e.g. day of week, hour, minute) and a date frequency.

    ex) df.pipe(tsheatmap, date='datetime', val='cont')
    '''

    timeunit = getattr(df[date].dt, unit)
    date = pd.Grouper(key=date, freq=freq)

    if val is None:
        a = df.groupby([date, timeunit]).size().unstack()
    else:
        a = df.groupby([date, timeunit])[val].mean().unstack()

    a.index = a.index.astype(str)
    sns.heatmap(a)
    plt.xlabel(unit)

def tslineplot(df, date, val=None, by=None, area=False, freq='M'):
    '''
    Creates a time series line plot of counts for a datetime variable.

    ex) df.pipe(tslineplot, date='datetime', by='cat')
    ex) df.pipe(tslineplot, date='datetime', val='cont', by='cat')
    ex) df.pipe(tslineplot, date='datetime', by=pd.qcut(df['cont'], 3))
    '''

    df = df.copy()

    kind = 'area' if area == True else 'line'

    date = pd.Grouper(key=date, freq=freq)

    if by is None and val is None:
        df.groupby(date).size().plot(kind=kind)
    elif by is None:
        df.groupby(date)[val].mean().fillna(0).plot(kind=kind)
    elif val is None:
        df.groupby([date, by]).size().unstack(by).plot(kind=kind)
    else:
        df.groupby([date, by])[val].mean().fillna(0).unstack(by).plot(kind=kind)

    if by:
        plt.legend(title=by, loc=(1, 0))

######################################
##### Model Performance Plotting #####
######################################
def plot_calibration_curve(model, X, y):
    '''
    Creates a plot of the calibration curve for a binary classification model.

    ex) plot_calibration_curve(model, xtest, ytest)
    '''

    prob_true, prob_pred = calibration_curve(y, model.predict_proba(X)[:, 1], n_bins=10)
    plt.plot(prob_pred, prob_true)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('Predicted Proportion')
    plt.ylabel('True Proportion')

def plot_classification_metrics(model, X, y, threshold=0.5):
    '''
    Creates plots of the f1 score, recall, and precision curves over several
    threshold values.

    ex) plot_classification_metrics(model, xtest, ytest)
    '''

    f1 = []
    recall = []
    precision = []

    for threshold in np.linspace(.1, 1, 10):
         pred = model.predict_proba(X)[:, 1] > threshold

         f1.append(f1_score(y, pred))
         recall.append(recall_score(y, pred))
         precision.append(precision_score(y, pred))

    plt.plot(np.linspace(.1, 1, 10), f1, label='f1')
    plt.plot(np.linspace(.1, 1, 10), recall, label='recall')
    plt.plot(np.linspace(.1, 1, 10), precision, label='precision')

    plt.legend(title='Metric', loc=(1, 0))

def plot_classification_report(model, X, y, threshold=0.5):
    '''
    Creates a heat map of the classification report for a model.

    ex) plot_classification_report(model, xtest, ytest)
    '''

    if len(y.value_counts()) > 2:
        pred = model.predict(X)

    else:
        try:
            pred = model.predict_proba(X)[:, 1] > threshold
        except:
            pred = model.predict(X)

    a = classification_report(y, pred)

    lines = a.split('\n')[2: -3]

    classes = []
    matrix = []
    for line in lines:
        s = line.split()
        classes.append(s[0])
        matrix.append([float(x) for x in s[1: -1]])

    df = pd.DataFrame(matrix, index=classes, columns=['precision', 'recall', 'f1'])

    sns.heatmap(df, annot=True, fmt='.2f')
    plt.ylabel('Class')

def plot_confusion_matrix(model, X, y, normalize=False, threshold=0.5):
    '''
    Creates a heat map of the confusion matrix for a binary or multiclass
    classification model.

    ex) plot_confusion_matrix(model, xtest, ytest, normalize='index')
    '''

    if len(y.value_counts()) > 2:
        pred = model.predict(X)

    else:
        try:
            pred = model.predict_proba(X)[:, 1] > threshold
        except:
            pred = model.predict(X)

    a = confusion_matrix(y, pred).astype(float)

    if normalize == 'index':
        a = np.divide(a, np.sum(a, 1).reshape(len(a), 1))
    elif normalize == 'column':
        a = np.divide(a, np.sum(a, 0))

    sns.heatmap(a, annot=True, fmt='.2f')
    plt.ylabel('True')
    plt.title('Predicted')

def plot_decision_tree(X, y, filename, directory='~', **kwargs):
    '''
    Creates a graphviz plot of a decision tree and saves it to a file.

    ex) plot_decision_tree(xtrain, ytrain, 'tree')
    '''

    if directory[0] == '~':
        directory = os.path.expanduser(directory)
    filename = os.path.join(directory, filename)

    model = DecisionTreeClassifier(**kwargs)
    model.fit(X, y)

    dot_data = export_graphviz(model, class_names=y.astype(str).unique(),
        feature_names=X.columns, filled=True, out_file=None, rounded=True)

    graph = graphviz.Source(dot_data)
    graph.render(filename)
    subprocess.call(('open', filename + '.pdf'))

def plot_learning_curves(model, X, y):
    '''
    Creates a plot of the sample size vs. the mean CV train set scores and of
    the sample size vs. the mean CV test set scores.

    ex) plot_learning_curves(model, xtrain, ytrain)
    '''

    cv = StratifiedKFold(n_splits=5, shuffle=True)
    sizes, train, validation = learning_curve(model, X, y, cv=cv, scoring='roc_auc')
    plt.plot(sizes, np.mean(train, axis=1), label='train')
    plt.plot(sizes, np.mean(validation, axis=1), label='validation')
    plt.xlabel('Sample Size')
    plt.ylabel('Performance')
    plt.legend(loc=(1, 0))

def plot_predicted_probabilities(model, X, y):
    '''
    Creates a density plot of the predicted probabilities for a model grouped by the
    target class.

    ex) plot_predicted_probabilities(model, xtest, ytest)
    '''

    df = cbind(y, model.predict_proba(X)[:, 1])
    df.pipe(distplot, by=df.columns[0], col=df.columns[1])

def plot_top_features(model, X, attr, n=10):
    '''
    Creates a bar plot of the feature importance scores of the top features for a model.

    ex) plot_top_features(model, xtrain, 'coef_')
    '''

    scores = feature_scores(model, X, attr, sort_abs=True)[:n]
    a = scores.set_index(0).sort_values(by='abs')[1]

    colors = ''.join(['g' if i >= 0 else 'r' for i in a])
    a.plot.barh(color=colors)
    plt.ylabel('')
    plt.title('Top Features')
