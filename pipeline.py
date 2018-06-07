from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class OneHotFeature(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X):
        d = pd.get_dummies(X[[self.column]], columns=[self.column])
        self.categories = pd.DataFrame(columns=d.columns)
        return self

    def transform(self, X):
        d = pd.get_dummies(X[[self.column]], columns=[self.column])
        d = self.categories.align(d, axis=1, join='left')[1]
        return d.fillna(0).astype(int)


class Select(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X):
        return self

    def transform(self, X):
        try:
            return X[self.columns]
        except KeyError:
            return pd.DataFrame()


class Aggregate(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            return X.groupby(self.columns).sum().reset_index()
        except KeyError:
            return pd.DataFrame(X.sum()).T
