from sklearn.base import TransformerMixin, _OneToOneFeatureMixin, BaseEstimator
import numpy as np


class AsinhScaler(_OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    def __init__(self, copy=False, with_median=True, with_mad=True, **kwargs):
        self.tab_to_transform_ = None
        self.to_transform_ = None
        self.median_ = None
        self.mad_ = None
        self.copy = copy
        self.with_median = with_median
        self.with_mad = with_mad

    def fit(self, X, *args, **kwargs):
        return self

    @staticmethod
    def transform(X, *args, **kwargs):
        X_t = X.copy()
        return np.log(X_t + np.sqrt(np.square(X_t) + 1))

    @staticmethod
    def inverse_transform(X, *args, **kwargs):
        X = (np.exp(X) - np.exp(-X)) / 2
        X_t = X.copy()
        return X_t


class AsinhMedianScaler(_OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    def __init__(self, copy=False, with_median=True, with_mad=True, **kwargs):
        self.tab_to_transform_ = None
        self.to_transform_ = None
        self.median_ = None
        self.mad_ = None
        self.copy = copy
        self.with_median = with_median
        self.with_mad = with_mad

    def fit(self, X, *args, **kwargs):
        self.median_ = np.median(X, axis=0)
        self.mad_ = np.median(np.abs(X - self.median_), axis=0)

        # Ignore where MAD = 0
        try:
            self.median_[self.mad_ == 0] = 0
            self.mad_[self.mad_ == 0] = 1.

        except TypeError: #If only a float:
            if self.mad_ == 0:
                self.mad_ = 1
                self.median_ = 0

        return self

    def transform(self, X, *args, **kwargs):
        X_t = (X - self.median_) / self.mad_
        return np.log(X_t + np.sqrt(np.square(X_t) + 1))

    def inverse_transform(self, X, *args, **kwargs):
        X = (np.exp(X) - np.exp(-X)) / 2
        X_t = X * self.mad_ + self.median_
        return X_t


class MedianScaler(_OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    def __init__(self, copy=False, with_median=True, with_mad=True, **kwargs):
        self.tab_to_transform_ = None
        self.to_transform_ = None
        self.median_ = None
        self.mad_ = None
        self.copy = copy
        self.with_median = with_median
        self.with_mad = with_mad

    def fit(self, X, *args, **kwargs):
        self.median_ = np.median(X, axis=0)
        self.mad_ = np.median(np.abs(X - self.median_), axis=0)

        # Ignore where MAD = 0
        try:
            self.median_[self.mad_ == 0] = 0
            self.mad_[self.mad_ == 0] = 1.

        except TypeError: #If only a float:
            if self.mad_ == 0:
                self.mad_ = 1
                self.median_ = 0

        return self

    def transform(self, X, *args, **kwargs):
        X_t = (X - self.median_) / self.mad_
        return X_t

    def inverse_transform(self, X, *args, **kwargs):
        X_t = X * self.mad_ + self.median_
        return X_t
