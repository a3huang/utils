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

def plot_interaction(df, col, by, val, kind='line'):
    '''
    Creates an interaction line plot or heat map between 2 predictor variables and
    a target variable.

    ex) df.pipe(plot_interaction, col='cat', by='cat', val='cont')
    '''

    a = df.pipe(table, col, by, val)

    if kind == 'heat':
        sns.heatmap(a, annot=True, fmt='.2f')
    elif kind == 'line':
        a.plot()

    plt.legend(loc=(1, 0))

def plot_binary_target(df, x, y):
    df.groupby(x)[y].mean().plot()

def plot_line_interaction(df, x1, x2, y):
    df.pipe(table, x1, x2, y).plot()
    plt.legend(loc=(1, 0))

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

def create_explainer(model, X):
    '''
    Convenience function for creating a LIME explainer object.

    ex) create_explainer(model, X_train)
    '''

    explainer = LimeTabularExplainer(X.values, feature_names=X.columns.values)
    return explainer

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
    gains = scoring_table(y, model.predict_proba(X)[:, 1])['Target Metrics',
        'Cumulative'].reset_index()
    gains.columns = ['Decile', 'Cumulative']

    deciles = np.append([0], gains['Decile'].values / 10.)
    gains = np.append([0], gains['Cumulative'].values)

    plt.plot(deciles, gains)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('Decile')
    plt.ylabel('Gain')

def plot_lift_curve(model, X, y):
    gains = scoring_table(y, model.predict_proba(X)[:, 1])['Target Metrics',
        'Cumulative'].reset_index()
    gains.columns = ['Decile', 'Cumulative']

    deciles = gains['Decile'].values / 10.
    gains = gains['Cumulative'].values

    plt.plot(deciles, gains/deciles)
    plt.xlabel('Decile')
    plt.ylabel('Lift')
