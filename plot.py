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
from statsmodels.graphics.mosaicplot import mosaic

import graphviz
import os
import pydotplus
import subprocess

from utils.data import *

# deprecate
def create_explainer(model, X):
    '''
    Convenience function for creating a LIME explainer object.

    ex) create_explainer(model, X_train)
    '''

    explainer = LimeTabularExplainer(X.values, feature_names=X.columns.values)
    return explainer
#####

##########################
##### Basic Plotting #####
##########################
#TODO: change API to by=['Sex', 'Cabin']?

def barplot(df, col, by=None, val=None, orient='v', prop=False, stacked=False):
    '''
    Creates a bar plot of counts for a categorical variable or of the mean for a
    continuous variable.

    ex) df.pipe(barplot, by='Survived', col='Sex')
    ex) df.pipe(barplot, by=pd.qcut(df.Age, 3), col=pd.qcut(df.Fare, 3))
    ex) df.pipe(barplot, col=df.date.dt.weekday)
    '''

    kind = 'barh' if orient == 'h' else 'bar'
    normalize = 'index' if prop else False

    if by is None:
        if val is None:
            df = df.groupby(col).size()
            df = df / len(df) if normalize else df
            df.plot(kind=kind)
        else:
            df.groupby(col)[val].mean().plot(kind=kind)

    else:
        if val is None:
            df.pipe(table, col, by, normalize=normalize).plot(kind=kind, stacked=stacked)
        else:
            df.pipe(table, col, by, val).plot(kind=kind)

def boxplot(df, by, col, orient='v', **kwargs):
    '''
    Creates a box plot for a continuous variable grouped by a categorical variable.

    ex) df.pipe(boxplot, by='Survived', col='Age')
    ex) df.pipe(boxplot, by=pd.qcut(df.Age, 3), col='Fare')
    ex) df.pipe(boxplot, by='Survived', hue=pd.qcut(df.Age, 3), col='Fare')
    '''

    color = sns.color_palette()[0]

    if orient == 'h':
        x, y = col, by
    else:
        x, y = by, col

    sns.boxplot(x=x, y=y, data=df, **kwargs)

def nice_bin_range(a, bin_mult=1, bin_range=None):
    # make temporary plot of min and max just to get the "right" x-axis tick marks
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

def histogram(df, col, by=None, nice=False, prop=False, **kwargs):
    '''
    Creates a histogram for a continuous variable.

    ex) df.pipe(histogram, col='Age')
    ex) df.pipe(histogram, by=pd.qcut(df.Age, 3), col='Fare')
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

def densityplot(df, col, by=None, range=None):
    '''
    Creates a density plot for a continuous variable.

    ex) df.pipe(densityplot, col='Age')
    ex) df.pipe(densityplot, by=pd.qcut(df.Age, 3), col='Fare')
    '''

    if by is None:
        sns.kdeplot(df[col])
        plt.legend().remove()

    else:
        for g, c in df.groupby(by)[col]:
            sns.kdeplot(c, label=g, shade=True)

        plt.xlim(range)
        plt.legend(loc=(1, 0))


def heatmap(df, row, col, val=None, normalize=False):
    '''
    Creates a heat map of counts for 2 categorical variables.

    ex) df.pipe(heatmap, row='Cabin', col='Sex', val='Survived')
    ex) df.pipe(heatmap, row=pd.qcut(df.Age, 3), col=pd.qcut(df.Fare, 3))
    '''

    if val:
        sns.heatmap(df.pipe(table, row, col, val), annot=True, fmt='.2f')
    else:
        sns.heatmap(df.pipe(table, row, col, normalize=normalize), annot=True, fmt='.2f')

def lineplot(df, x, y, by=None, **kwargs):
    '''
    Creates a line plot of the mean value for a continuous variable.

    ex) df.pipe(lineplot, x='Cabin', y='Survived', by='Sex')
    ex) df.pipe(lineplot, x=pd.cut(df.Age, 3), y='Survived')
    ex) df.pipe(lineplot, x=df.date.dt.weekday, y='Amount')
    '''

    df = df.copy()

    if by is not None:
        if not isinstance(by, str):
            df[by.name] = by
            by = by.name

    sns.pointplot(x, y, hue=by, data=df, ci=False, **kwargs)
    plt.legend(title=by, loc=(1, 0))

def scatterplot(df, x, y, by=None, **kwargs):
    '''
    Creates a scatter plot for 2 continuous variables.

    ex) df.pipe(scatterplot, x='Age', y='Fare', by='Survived')
    '''

    df = df.copy()

    if by is not None:
        if not isinstance(by, str):
            df[by.name] = by
            by = by.name

    sns.lmplot(x, y, hue=by, data=df, **kwargs)
    plt.legend(title=by, loc=(1, 0))

################################
##### Time Series Plotting #####
################################
def tslineplot(df, date, by=None, val=None, area=False, freq='M'):
    '''
    Creates a time series line plot of counts for a date variable.

    ex) df.pipe(tslineplot, date='date', by='diet')
    ex) df.pipe(tslineplot, date='date', by=pd.qcut(df['purchase_frequency'], 3))
    '''

    df = df.copy()

    if by is not None:
        if not isinstance(by, str):
            df[by.name] = by
            by = by.name

    if area:
        kind = 'area'
    else:
        kind = 'line'

    date = pd.Grouper(key=date, freq=freq)

    if by and val:
        df.groupby([date, by])[val].mean().unstack(by).plot(kind=kind)
    elif by:
        df.groupby([date, by]).size().unstack(by).plot(kind=kind)
    elif val:
        df.groupby(date)[val].mean().plot(kind=kind)
    else:
        df.groupby(date).size().plot(kind=kind)

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

def plot_decision_tree(X, y, filename, directory='/Users/alexhuang', **kwargs):
    '''
    Creates a graphviz plot of a decision tree and saves it to a file.

    ex) plot_decision_tree(X, y, 'tree')
    '''

    filename = os.path.join(directory, filename)

    model = DecisionTreeClassifier(**kwargs)
    model.fit(X, y)

    dot_data = export_graphviz(model, class_names=y.astype(str).unique(),
        feature_names=X.columns, filled=True, out_file=None, rounded=True)

    graph = graphviz.Source(dot_data)
    graph.render(filename)
    subprocess.call(('open', os.path.join(filename, '.pdf')))

def plot_predicted_probabilities(model, X, y):
    '''
    Creates a density plot of the predicted probabilities for a model grouped by the
    target class.

    ex) plot_predicted_probabilities(model, xtest, ytest)
    '''

    df = cbind(y, model.predict_proba(X)[:, 1])
    df.pipe(distplot, by=df.columns[0], col=df.columns[1])
#####

def multicol_heatplot(df, by, cols):
    '''
    Creates a heat map of the average values of several continuous variables
    grouped by the given categorical variable. Automatically standardizes
    the continuous variables to faciliate comparison.

    ex) df.pipe(multicol_heatplot, 'Type', ['HP', 'Attack', 'Defense'])
    '''

    s = MinMaxScaler()

    a = pd.DataFrame(s.fit_transform(df.fillna(0)[cols]), columns=cols)
    a = cbind(a, df[by])
    a = a.groupby(by)[cols].mean()
    sns.heatmap(a, annot=True, fmt='.2f')
    plt.xticks(rotation=90)

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

# deprecate
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

#deprecate
def plot_2d_projection(df, by, method=None, sample_size=None):
    '''
    Creates a scatter plot of the 2-D projection of the dataset. Groups by a
    categorical variable using color. Uses PCA method by default for
    dimensionality reduction.

    ex) df.pipe(plot_2d_projection, by='Legendary')

    ex) projection = make_pipeline(StandardScaler(), PCA())
        df1 = cbind(df, projection.fit_transform(df.drop('Survived', 1))[:, :1])
        df1.pipe(scatterplot, x='PCA1', y='PCA2', by='Survived')
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

def plot_roc_curves2(models, X, y):
    '''
    Creates a plot of the roc curve for each binary classification model pipeline.

    ex) plot_roc_curves([model1, model2, model3], xtest, ytest)
    '''

    models = models if isinstance(models, list) else [models]

    for i, model in enumerate(models):
        pred = model.predict_proba(X)[:, 1]
        false_pos_rate, true_pos_rate, _ = roc_curve(y, pred)
        plt.plot(false_pos_rate, true_pos_rate, label=i)

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    if len(models) > 1:
        plt.legend(title='Model', loc=(1, 0))

# deprecate
def plot_roc_curves(model, X, y, label=1):
    '''
    Creates a line plot of the roc curve for the given model and data.

    ex) plot_roc_curves(model, xtest, ytest)
    '''

    true = y.pipe(dummy).iloc[:, label]
    pred = model.predict_proba(X)[:, label]

    fpr, tpr, _ = roc_curve(true, pred)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

# deprecate
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

def plot_gains_curve(model, X, y):
    gains = scoring_table(y, model.predict_proba(X)[:, 1])['Target Metrics', 'Cumulative'].reset_index()
    gains.columns = ['Decile', 'Cumulative']

    deciles = np.append([0], gains['Decile'].values / 10.)
    gains = np.append([0], gains['Cumulative'].values)

    plt.plot(deciles, gains)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('Decile')
    plt.ylabel('Gain')

def plot_lift_curve(model, X, y):
    gains = scoring_table(y, model.predict_proba(X)[:, 1])['Target Metrics', 'Cumulative'].reset_index()
    gains.columns = ['Decile', 'Cumulative']

    deciles = gains['Decile'].values / 10.
    gains = gains['Cumulative'].values

    plt.plot(deciles, gains/deciles)
    plt.xlabel('Decile')
    plt.ylabel('Lift')
