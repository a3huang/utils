import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lifelines import KaplanMeierFitter
from lime.lime_tabular import LimeTabularExplainer
from pandas.api.types import is_numeric_dtype, is_string_dtype
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, f1_score, \
        recall_score, precision_score
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, export_graphviz

import graphviz
import os
import pydotplus
import subprocess

from utils.data import *

def dummy_categorical(df, n, shuffle=False):
    '''
    Creates a categorical variable with labels 1 to n for testing purposes.

    ex) df['dummy'] = df.pipe(dummy_categorical, 5)
    '''

    size = df.shape[0] / n
    remainder = df.shape[0] - n * size

    a = []
    a.append(np.full((size + remainder, 1), 1))

    for i in range(2, n+1):
        a.append(np.full((size, 1), i))

    b = pd.DataFrame(np.append(a[0], a[1:]))

    if shuffle:
        b = np.random.permutation(b)

    return b

def dummy_continuous(df, loc=0, scale=1):
    '''
    Creates a continuous variable from a normal distribution for testing purposes.

    ex) df['dummy'] = df.pipe(dummy_continuous)
    '''

    return pd.DataFrame(np.random.normal(loc=loc, scale=scale, size=df.shape[0]))

def pipeline_nth_step(pipeline, X, n):
    '''
    Returns the resulting dataframe after applying the nth step of an sklearn
    pipeline.

    ex) pipeline_nth_step(pipeline, X_train, 5)
    '''

    a = pipeline.steps[0][1].transform(X.values)
    for i in range(1, n):
        a = pipeline.steps[i][1].transform(a)
    return pd.DataFrame(a, columns=X.columns)

def facet(df, row, col, **kwargs):
    '''
    Convenience function for creating seaborn facet grids.

    ex) df.pipe(facet, row='Type 1', col='Type 2').map(plt.scatter, 'Attack', 'Defense')
    '''

    return sns.FacetGrid(df, row=row, col=col, **kwargs)

def create_explainer(model, X):
    '''
    Convenience function for creating a LIME explainer object.

    ex) create_explainer(model, X_train)
    '''

    explainer = LimeTabularExplainer(X.values, feature_names=X.columns.values)
    return explainer

def ceil_with_base(x, base):
    '''
    Takes the ceiling of a number with respect to any base.

    ex) ceil_with_base(0.03, 0.05) -> 0.05
    ex) ceil_with_base(0.02, 0.05) -> 0.05
    '''

    return base * np.ceil(float(x) / base)

def round_with_base(x, base):
    '''
    Rounds a number with respect to any base.

    ex) round_with_base(0.03, 0.05) -> 0.05
    ex) round_with_base(0.02, 0.05) -> 0
    '''

    return base * np.round(float(x) / base)

def nice_round(x):
    '''
    Rounds a number "nicely" to the nearest denomination of 5 or 10. Numbers less
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

def truncate(x):
    '''
    Truncates a decimal to its first nonzero digit.

    ex) truncate(0.0345) -> 0.03
    '''

    if x == 0:
        return 0
    sign = -1 if x < 0 else 1
    scale = int(-np.floor(np.log10(abs(x))))
    if scale <= 0:
        scale = 1
    factor = 10**scale
    return sign * np.floor(abs(x) * factor) / factor

def take(iterator, n):
    '''
    Returns the nth item from an iterator.

    ex) take(df.groupby('Generation'), 2)
    '''

    for i, a in enumerate(iterator):
        if i != n:
            continue
        else:
            return a

def nice_range_bin(ax, range=None):
    '''
    Helper function for matplotlib histograms to find the right range and number
    of bins so that the bar edges will line up nicely with existing x-axis tick
    marks.

    ex) fig, ax = plt.subplots()
        df[col].plot.hist(ax=ax)
        range, bins = nice_range_bin(ax)
    '''

    ticks = ax.get_xticks()

    if range is None:
        range = ticks[1], ticks[-2]

    total_length = range[1] - range[0]
    bin_size = ticks[1] - ticks[0]
    bins = int(total_length / bin_size)
    return range, bins

def nice_hist(df, col, range=None, prop=False):
    '''
    Creates a "nice" histogram by drawing the default histogram first and then
    adjusting the bin edges so that they line up with the existing x-axis
    tick marks.

    This function takes advantage of the fact that matplotlib's
    plt.hist already comes up with "nice" values for the x-axis tick marks.

    Note: Does not play nicely with seaborn's FacetGrid.
    '''

    fig, ax = plt.subplots()
    df[col].plot.hist(range=range, ax=ax)
    range, bins = nice_range_bin(ax, range)
    plt.clf()

    if prop:
        weights = np.ones_like(df[col]) / float(len(df[col]))
    else:
        weights = None

    df[col].plot.hist(range=range, bins=bins, weights=weights, alpha=0.4)

def nice_hist_approx(a, bins=10, **kwargs):
    '''
    Creates a "nice" histogram by adjusting both the bin edges and the x-axis tick
    marks so that they line up nicely.

    Unlike nice_hist, this function uses custom functions to determine "nice"
    locations for the x-axis tick marks. This leads to tick values that are not
    as nice as the ones matplotlib's plt.hist gives.

    Note: Does not play nicely with seaborn's FacetGrid.
    '''

    heights, edges = np.histogram(a)
    width = (edges[-1] - edges[0]) / bins

    if width < 1:
        width = truncate(width)
    else:
        width = nice_round(width)

    new_edges = [round_with_base(edges[0], width) + i*width for i in range(len(edges))]
    plt.hist(a, bins=new_edges, **kwargs)
    plt.xticks(new_edges)

def prop_hist(a, prop=False, **kwargs):
    '''
    Creates a histogram where each bar displays the proportion of observations
    in each bin rather than their counts.
    '''

    if prop:
        weights = np.ones_like(a) / float(len(a))
    else:
        weights = None

    plt.hist(a, weights=weights, **kwargs)

def barplot(df, col, by=None, kind=None, prop=False):
    '''
    Creates a bar plot for a categorical variable. Group by an optional 2nd
    categorical variable.

    ex) df.pipe(barplot, col='HP', by='Type')
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

def boxplot(df, col, by, facet_by=None, sort_median=False):
    '''
    Creates a grouped box plot for a continuous variable. Facet by an optional
    3rd categorical variable.

    ex) df.pipe(boxplot, col='HP', by='Type')
    ex) df.pipe(boxplot, col='Amount', by='Category', facet_by='Target')
    '''

    if sort_median:
        order = df.groupby(by)[col].median().sort_values().index
    else:
        order = None

    if facet_by:
        g = sns.FacetGrid(df, col=facet_by)
        g.map(sns.boxplot, by, col, order=order)
    else:
        sns.boxplot(x=col, y=by, data=df, order=order, orient='h')

def distplot(df, col, by=None, prop=False, facet=False, range=None):
    '''
    Creates a histogram for a continuous variable. Group by an optional 2nd
    categorical variable.

    ex) df.pipe(distplot, col='HP', by='Type')
    '''

    if by:
        if facet:
            g = sns.FacetGrid(df, col=by)
            g.map(prop_hist, col, range=range)
            r, b = nice_range_bin(g.axes.flat[0], range)
            plt.clf()

            g = sns.FacetGrid(df, col=by, col_wrap=3)
            g.map(prop_hist, col, prop=prop, range=r, bins=b, alpha=0.4)
        else:
            for group, column in df.groupby(by)[col]:
                sns.kdeplot(column, label=group, shade=True)
            plt.xlim(range)
            plt.legend(title=by, loc=(1, 0))

    else:
        nice_hist(df, col, prop=prop, range=range)

def heatplot(df, x, y, z=None, normalize=False):
    '''
    Creates a heat map between 2 categorical variables. Calculate the mean for an
    optional 3rd continuous variable.

    ex) df.pipe(heatplot, x='Type 1', y='Type 2', z='Attack')
    '''

    if z:
        sns.heatmap(df.pipe(table, x, y, z), annot=True, fmt='.2f')
    else:
        sns.heatmap(df.pipe(table, x, y, normalize=normalize), annot=True, fmt='.2f')

def multicol_heatplot(df, by, cols):
    '''
    Creates a heat map of the average values of several continuous variables
    grouped by the given categorical variable. Automatically standardizes
    the continuous variables to faciliate comparison.

    ex) df.pipe(multicol_heatplot, 'Legendary', ['HP', 'Attack', 'Defense'])
    '''

    s = MinMaxScaler()

    a = pd.DataFrame(s.fit_transform(df.fillna(0)[cols]), columns=cols)
    a = cbind(a, df[by])
    a = a.groupby(by)[cols].mean()
    sns.heatmap(a, annot=True, fmt='.2f')
    plt.xticks(rotation=90)

def scatplot(df, x, y, by=None, facet=False):
    '''
    Creates a scatter plot for 2 continuous variables. Group by an optional 3rd
    categorical variable.

    ex) df.pipe(scatplot, x='Attack', y='Defense', by='Legendary')
    '''

    if by:
        if facet:
            g = sns.FacetGrid(df, col=by)
            g.map(sns.regplot, x, y, fit_reg=False, ci=False)
        else:
            sns.lmplot(x=x, y=y, hue=by, data=df, legend=False, fit_reg=False, ci=False)
            plt.legend(title=by, loc=(1, 0))
    else:
        sns.lmplot(x=x, y=y, hue=by, data=df, legend=False, fit_reg=False, ci=False)

def tsplot(df, date, by=None, val=None, freq='M', area=False):
    '''
    Creates a time series plot of counts for a date variable or of mean values
    for a continuous variable. Group by an optional 3rd categorical variable.

    ex) df.pipe(tsplot, date='date', by='Category')
    '''

    if area:
        kind = 'area'
    else:
        kind = 'line'

    if by and val:
        df.groupby([pd.Grouper(key=date, freq=freq), by])[val].mean().unstack(by).plot(kind=kind)
    elif by:
        df.groupby([pd.Grouper(key=date, freq=freq), by]).size().unstack(by).plot(kind=kind)
    elif val:
        df.groupby(pd.Grouper(key=date, freq=freq))[val].mean().plot(kind=kind)
    else:
        df.groupby(pd.Grouper(key=date, freq=freq)).size().plot(kind=kind)

    if by:
        plt.legend(title=by, loc=(1, 0))

def tsboxplot(df, date, col, freq='M'):
    '''
    Creates a time series box plot for a continuous variable.

    ex) df.pipe(tsboxplot, date='date', col='Amount')
    '''

    groups = df.groupby(pd.Grouper(key=date, freq=freq))

    columns = []
    for i, g in groups:
        a = pd.DataFrame(g[col]).reset_index(drop=True).rename(columns={col: i})
        columns.append(a)

    data = pd.concat(columns, axis=1)
    sns.boxplot(data=data, orient='h')
    plt.xlabel(col)

def timeunit_barplot(df, date, unit='weekday', col=None):
    '''
    Creates a bar plot of counts or the average of continuous variable grouped
    by a unit of time (e.g. minute, hour, day, weekday)

    ex) df.pipe(timeunit_barplot, date='date', unit='weekday', col='Total')
    '''

    df['unit'] = df[date].pipe(time_unit, unit)

    if col:
        df.groupby('unit')[col].mean().plot.bar()
        plt.ylabel('')
    else:
        df['unit'].value_counts().sort_index().plot.bar()
        plt.ylabel(col)

    plt.xlabel(unit)

def timeunit_heatplot(df, date, freq='M', unit='weekday', col=None):
    '''
    Creates a heat plot of counts or the average of a continuous variable
    grouped by a unit of time (e.g. minute, hour, day, weekday) on the y-axis
    and a frequency on the x-axis.

    ex) df.pipe(timeunit_heatplot, date='date', col='Total')
    '''

    df = df.copy()
    df['unit'] = df[date].pipe(time_unit, unit)

    if col:
        a = df.groupby(['unit', pd.Grouper(key=date, freq=freq)])[col].mean().unstack()
    else:
        a = df.groupby(['unit', pd.Grouper(key=date, freq=freq)]).size().unstack()

    sns.heatmap(a)

    new_labels = [i.get_text().split('T')[0] for i in plt.gca().get_xticklabels()]
    plt.gca().set_xticklabels(new_labels)
    plt.ylabel(unit)

def generate_distribution_plots(df, by, folder_name, omit=None,
                                default_dir='/Users/alexhuang/'):
    '''
    Generates a box plot for each continuous variable and a bar plot for each
    categorical variable that are grouped by the given categorical variable.

    ex) df.pipe(generate_distribution_plots, by='Target', folder_name='plots')
    '''

    df = df.copy()

    if omit:
        df = df.drop(omit, 1)

    directory = default_dir + folder_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i, col in enumerate(df.drop(by, 1)):
        if is_numeric_dtype(df[col]):
            df.pipe(boxplot, by=by, col=col)

        elif is_string_dtype(df[col]):
            if len(df[col].value_counts()) > 100:
                print '[WARNING]: %s has too many categories' % col
                continue
            df.pipe(barplot, by=by, col=col)

        plt.xlabel('')
        plt.title(col)
        plt.savefig(directory + '%s.png' % i, bbox_inches="tight")
        plt.close()
        print 'Saved Plot: %s' % col

def generate_partial_dependence_plots(df, target, folder_name, omit=None,
                                      default_dir='/Users/alexhuang/'):
    '''
    Fits a Logistic GAM and generates the partial dependence plot for each
    variable against the target variable.

    ex) df.pipe(generate_partial_dependence_plots, target='Target',
                folder_name='plots')
    '''

    df = df.copy()

    if omit:
        df = df.drop(omit, 1)

    X = df[df.columns.difference([target])]
    y = df[target]

    directory = default_dir + folder_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    gam = LogisticGAM().gridsearch(X, y)
    grid = generate_X_grid(gam)

    for i in range(len(X.columns)):
        p = gam.partial_dependence(grid, feature=i)
        plt.plot(grid[:, i], p)
        plt.title(X.columns[i])
        plt.savefig(directory + '%s.png' % i, bbox_inches="tight")
        plt.close()
        print 'Saved Plot: %s' % col

def plot_interaction(df, col, by, val, kind='line'):
    '''
    Creates an interaction line plot or heat map between 2 predictor variables and
    a 3rd target variable.

    ex) df.pipe(plot_interaction, col='Type 1', by='Type 2', val='Attack')
    '''

    a = df.pipe(table, col, by, val)

    if kind == 'box':
        sns.boxplot(x=col, y=val, hue=by, data=df)
    elif kind == 'heat':
        sns.heatmap(a, annot=True, fmt='.2f')
    elif kind == 'line':
        a.plot()

    plt.legend(title=by, loc=(1, 0))

def plot_2d_projection(df, by, method=None, sample_size=None):
    '''
    Creates a scatter plot of the 2-D projection of the dataset. Groups by a
    categorical variable using color. Uses PCA method by default for
    dimensionality reduction.

    ex) df.pipe(plot_2d_projection, by='Legendary')
    '''

    df = df.copy()

    if method is None:
        method = PCA()

    pipeline = make_pipeline(StandardScaler(), method)

    if sample_size:
        df = df.sample(sample_size)

    X = df.drop(by, 1)
    df['pca1'] = pipeline.fit_transform(X)[:, 0]
    df['pca2'] = pipeline.fit_transform(X)[:, 1]

    df.pipe(scatplot, x='pca1', y='pca2', by=by)

def plot_class_metrics(model, X, y, label=1):
    '''
    For each threshold value, calculates the mean 5-fold CV f1, recall, and
    precision scores. Creates a line plot of the threshold values vs. each of
    the metrics.

    ex) plot_class_metrics(model, xtest ytest)
    '''

    outer_f1 = []
    outer_recall = []
    outer_precision = []
    for threshold in np.linspace(.1, 1, 10):
         true = pd.Series(y).pipe(dummy).iloc[:, label]
         pred = model.predict_proba(X)[:, label] > threshold

         outer_f1.append(f1_score(true, pred))
         outer_recall.append(recall_score(true, pred))
         outer_precision.append(precision_score(true, pred))

    plt.plot(np.linspace(.1, 1, 10), outer_f1, label='f1')
    plt.plot(np.linspace(.1, 1, 10), outer_recall, label='recall')
    plt.plot(np.linspace(.1, 1, 10), outer_precision, label='precision')
    plt.legend(title='Metric', loc=(1, 0))

def plot_classification_report(model, X, y, threshold=0.5):
    '''
    Creates a heat map of the classification report for the given model.

    ex) plot_classification_report(model, xtest, ytest)
    '''

    if len(y.value_counts()) > 2:
        pred = model.predict(X)
    else:
        pred = model.predict_proba(X)[:, 1] > threshold

    a = classification_report(y, pred)
    lines = a.split('\n')[2:-3]

    classes = []
    matrix = []
    for line in lines:
        s = line.split()
        classes.append(s[0])
        matrix.append([float(x) for x in s[1:-1]])

    df = pd.DataFrame(matrix, index=classes, columns=['precision', 'recall', 'f1'])

    sns.heatmap(df, annot=True, fmt='.2f')
    plt.ylabel('Class')

def plot_confusion_matrix(model, X, y, threshold=0.5, normalize=False, label=1):
    '''
    Creates a heat map of the confusion matrix for the given model and data.

    ex) plot_confusion_matrix(model, xtest, ytest, normalize='index')
    '''

    true = y.pipe(dummy).iloc[:, label]

    try:
        pred = model.predict_proba(X)[:, label] > threshold
    except:
        pred = pd.Series(model.predict(X)).pipe(dummy).iloc[:, label]

    a = confusion_matrix(y, pred).astype(float)

    if normalize == 'index':
        a = np.divide(a, np.sum(a, 1).reshape(2, 1))
    elif normalize == 'column':
        a = np.divide(a, np.sum(a, 0))

    sns.heatmap(a, annot=True, fmt='.2f')
    plt.ylabel('True')
    plt.title('Predicted')


def plot_multi_confusion_matrix(model, X, y, normalize=False):
    '''
    Creates a heat map of the multilabel confusion matrix for the given model
    and data.

    ex) plot_multi_confusion_matrix(model, xtest, ytest, normalize='index')
    '''

    true = y
    pred = model.predict(X)

    a = confusion_matrix(true, pred).astype(float)

    if normalize == 'index':
        a = np.divide(a, np.sum(a, 1).reshape(len(a), 1))
    elif normalize == 'column':
        a = np.divide(a, np.sum(a, 0))

    sns.heatmap(a, annot=True, fmt='.2f')
    plt.ylabel('True')
    plt.title('Predicted')

def plot_roc_curves(model, X, y, label=1):
    '''
    Creates a line plot of the roc curve for the given model and data.

    ex) plot_roc_curves(model, xtest, ytest)
    '''

    true = y.pipe(dummy).iloc[:, label]

    try:
        pred = model.predict_proba(X)[:, label]
    except:
        pred = pd.Series(model.predict(X)).pipe(dummy).iloc[:, label]

    fpr, tpr, _ = roc_curve(true, pred)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

def compare_data_roc_curves(model, datasets, target, omit=None,
                            random_state=42):
    '''
    Compares the ROC curves for a given model over several datasets.

    ex) compare_data_roc_curves(model, [df1, df2, df3, df4, df5], target='cancel',
            omit=['user_id'])
    '''

    if omit is None:
        omit = []

    for i, df in enumerate(datasets, 1):
        X = df.drop(omit + [target], 1)
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
            random_state=random_state)

        model.fit(X_train, y_train)

        pred = model.predict_proba(X_test)[:, 1]
        true = y_test

        fpr, tpr, _ = roc_curve(true, pred)
        plt.plot(fpr, tpr, label=i)

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(title='dataset', loc=(1, 0))

def plot_learning_curves(model, X, y):
    '''
    For each sample size, splits the given data into 5 CV train test pairs.
    Calulates the mean CV score over the 5 training sets and the mean CV score
    over the 5 validation sets. Creates a line plot of the sample sizes vs.
    the mean CV train set scores and the mean CV validation set scores.

    ex) plot_learning_curves(model, xtrain ytrain)
    '''

    cv = StratifiedKFold(n_splits=5, shuffle=True)
    sizes, train, validation = learning_curve(model, X, y, cv=cv, scoring='roc_auc')
    plt.plot(sizes, np.mean(train, axis=1), label='train')
    plt.plot(sizes, np.mean(validation, axis=1), label='validation')
    plt.xlabel('Sample Size')
    plt.ylabel('Performance')
    plt.legend(loc=(1, 0))

def plot_calibration_curve(model, X, y):
    '''
    Creates a line plot of the calibration curve for the given model and data.

    ex) plot_calibration_curve(model, xtest, ytest)
    '''

    fp, mv = calibration_curve(y, model.predict_proba(X)[:, 1], n_bins=10)
    plt.plot(mv, fp)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('Predicted Proportion')
    plt.ylabel('True Proportion')

def plot_score_dists(model, X, y):
    '''
    Creates a density plot of model scores grouped by the target class. Useful
    for seeing how well the model separates out the target classes.

    ex) plot_score_dists(model, xtest, ytest)
    '''

    df = cbind(y, model.predict_proba(X)[:, 1]).rename(columns={0: 'score'})
    df.pipe(distplot, by=df.columns[0], col='score')

def plot_top_features(model, X, attr, n=10, label=0):
    '''
    Creates a bar plot of the top feature importance scores assigned by the
    given model.

    ex) plot_top_features(model, X_train, 'coef_')
    '''

    scores = feature_scores(model, X, attr, sort_abs=True, label=label)[:n]
    a = scores.set_index(0).sort_values(by='abs')[1]

    colors = ''.join(['g' if i >= 0 else 'r' for i in a])
    a.plot.barh(color=colors)
    plt.ylabel('')
    plt.title('Top Features')

def plot_explanations(explainer, model, X, i=None):
    '''
    Creates a bar plot of the top feature effects via LIME for the given row in
    the data. If no row number is specified, the function will pick one of the
    rows at random.

    ex) plot_explanations(explainer, model, X_test)
    '''

    if i is None:
        i = np.random.randint(0, X.shape[0])

    i = int(i)

    explanation = explainer.explain_instance(X.values[i], model.predict_proba)

    a = pd.DataFrame(explanation.as_list()).sort_index(ascending=False).set_index(0)[1]
    colors = ''.join(['g' if i >= 0 else 'r' for i in a])
    a.plot.barh(color=colors)
    plt.legend().remove()
    plt.ylabel('')

def plot_decision_tree(X, y, file_name, default_dir='/Users/alexhuang/',
                       max_depth=10, min_samples_leaf=100, **kwargs):
    '''
    Generates and saves a graphviz plot of a decision tree to a file and
    automatically opens the file for convenience.

    ex) plot_decision_tree(X, y, 'tree')
    '''

    file_name = default_dir + file_name

    model = DecisionTreeClassifier(max_depth=max_depth,
                                   min_samples_leaf=min_samples_leaf,
                                   **kwargs)
    model.fit(X, y)

    dot_data = export_graphviz(model,
                               out_file=None,
                               feature_names=X.columns,
                               filled=True, rounded=True,
                               class_names=y.astype(str).unique())

    graph = graphviz.Source(dot_data)
    graph.render(file_name)
    subprocess.call(('open', file_name + '.pdf'))

def plot_survival_curves(df, time, event, by=None):
    '''
    Creates survival curves grouped by the given categorical variable.

    ex) df.pipe(plot_survival_curves, time='days', event='cancel', by='state')
    '''

    df = df.copy()
    fig, ax = plt.subplots()
    kmf = KaplanMeierFitter()

    if by:
        a = df[by].dropna().astype(str)

        for i in a.unique():
            T = df.loc[df[by] == i, time]
            E = df.loc[df[by] == i, event]
            kmf.fit(T, event_observed=E, label=i)
            kmf.survival_function_.plot(ax=ax)

        plt.legend(title=by, loc=(1, 0))

    else:
        T = df[time]
        E = df[event]
        kmf.fit(T, event_observed=E)
        kmf.survival_function_.plot(ax=ax)
        plt.legend().remove()
