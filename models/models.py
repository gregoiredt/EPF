import pandas as pd
from sklearn.base import TransformerMixin, _OneToOneFeatureMixin, BaseEstimator, clone
import numpy as np
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso, QuantileRegressor
from functools import partial
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.model_selection._split import _BaseKFold
from skgarden import RandomForestQuantileRegressor
from sklearn.linear_model import QuantileRegressor, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from conformal.cqr import helper
import pickle
import os
import itertools

DIRECTORY = os.getcwd()


class SklearnModelPipeline:
    ''' Wrapper for scikit-learn type models trained on multiple hours

    '''

    def __init__(self, name, pipeline, features, max_train_size=None):
        ''' Initialisation

        Parameters
        ----------
        name: str, model name
        pipeline: BaseEstimator or dict, type of model to train. If dict, gives the type of model to train for each hour
        features: list, list of variables to train on
        max_train_size: None or int, maximum number of instances to train on
        '''

        self.max_train_size = max_train_size
        self.name = name
        self.features = features
        if isinstance(pipeline, dict):
            self.pipeline = pipeline.copy()
        else:
            self.pipeline = {h: clone(pipeline) for h in range(24)}
        self.trained_pipeline = {}

    def get_model(self, hour, trained=False):
        '''
        Returns the model for a given hour, either trained or not.

        Parameters
        ----------
        hour: int, between (0, 23)
        trained: boolean, whether to return the trained model or not

        Returns
        ---------

        pipeline: BaseEstimator,
        '''
        if trained:
            return self.trained_pipeline[hour]
        return self.pipeline[hour]

    def train(self, X, y, hour=0):
        '''
        Trains the models for a given hour

        Parameters
        ----------
        X: array, Training instances
        y: array, target instances
        hour: int, between (0, 23)

        Returns
        -------

        self
        '''
        model = clone(self.pipeline[hour])
        if not (self.max_train_size is None):
            trained_model = model.fit(X[-self.max_train_size:], y[-self.max_train_size:])
        else:
            trained_model = model.fit(X, y)
        self.trained_pipeline[hour] = trained_model

        return self

    def predict(self, X, hour=0, quantile=None):
        model = self.trained_pipeline[hour]
        if quantile is None:
            return model.predict(X)
        else:
            return model.predict(X, quantile=quantile)

    def get_feature_importance(self, hour, method='MDA', df_val=None, n_repeats=10, seed=42):
        """ Computes the feature importance for the differents models

        Parameters
        ----------

        method: str, Type of method used to compute the feature importance, (either "MDA", "MDI" or "coef").
        df_val:
        n_repeats:
        seed:

        Returns
        -------


        """
        feature_importances = pd.DataFrame()
        computed_importances = None
        if method == 'MDA':
            X_val = df_val.loc[df_val.Hour == hour, self.features]
            y_val = df_val.loc[df_val.Hour == hour, 'SpotPrice']
        model = self.trained_pipeline[hour]

        if method == 'MDA':
            result = permutation_importance(model, X_val, y_val, n_repeats=n_repeats, random_state=seed)
            computed_importances = result.importances_mean

        if method == 'MDI':
            if isinstance(model.named_steps['rgr'], GridSearchCV):
                computed_importances = model.named_steps['rgr'].best_estimator_.feature_importances_
            else:
                computed_importances = model.named_steps['rgr'].feature_importances_

        if method == 'coef':
            if isinstance(model.named_steps['rgr'], GridSearchCV):
                computed_importances = model.named_steps['rgr'].best_estimator_.coef_
            else:
                computed_importances = model.named_steps['rgr'].coef_

        df_res = pd.DataFrame({
            'feature': self.features,
            'importance': computed_importances
        })
        df_res['Hour'] = hour
        feature_importances = feature_importances.append(df_res)

        return feature_importances

    def _save_model(self, id_=None):
        """
        DEPRECATED
        :param id_:
        :return:
        """
        dirname = f'{self.name}' if id_ is None else f'{self.name}_{id_}'
        path = 'list_models/'
        dirpath = os.path.join(path, dirname)
        isdir = os.path.isdir(dirpath)
        if not isdir:
            os.mkdir(os.path.join(dirpath))
        for h in range(24):
            filename = os.path.join(dirpath, f'hour_{h}.pkl')
            # if isinstance(self.trained_pipeline[h], QuantileModelGatherer):
            #     ndirpath = os.path.join(dirpath, f'hour_{h}')
            #     self.trained_pipeline[h].save(ndirpath)

            with open(filename, 'wb') as file:
                pickle.dump(self.trained_pipeline[h], file)
        print('Successfully saved the model !')

    def _load_model(self, id_=None):
        """
        DEPRECATED
        :param id_:
        :return:
        """
        dirname = f'{self.name}' if id_ is None else f'{self.name}_{id_}'
        path = 'list_models/'
        dirpath = os.path.join(path, dirname)
        isdir = os.path.isdir(dirpath)
        assert isdir, 'This model has not been saved'

        for h in range(24):
            filename = os.path.join(dirpath, f'hour_{h}.pkl')
            # if isinstance(self.trained_pipeline[h], QuantileModelGatherer):
            #     ndirpath = os.path.join(dirpath, f'hour_{h}')
            #     self.trained_pipeline[h].load(ndirpath)

            with open(filename, 'rb') as file:
                self.trained_pipeline[h] = pickle.load(file)
        print('Successfully loaded the model !')


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.feature_names]

    def get_params(self, deep=True):
        return {'feature_names': self.feature_names}


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

        except TypeError:  # If only a float:
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

        try:
            self.median_[self.mad_ == 0] = 0
            self.mad_[self.mad_ == 0] = 1.

        except TypeError:  # If only a float:
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


class XGBAdaptater(XGBRegressor):
    def __init__(self, **kwargs):
        super(XGBAdaptater, self).__init__(**kwargs)

    def transform(self, X):
        return self.predict(X).reshape(-1, 1)


class LinearAdaptater(Lasso):
    def __init__(self, **kwargs):
        super(LinearAdaptater, self).__init__(**kwargs)

    def transform(self, X):
        return self.predict(X).reshape(-1, 1)


class QuantileLassoAdaptater(QuantileRegressor):
    def __init__(self, **kwargs):
        super(QuantileLassoAdaptater, self).__init__(**kwargs)

    def transform(self, X):
        return self.predict(X).reshape(-1, 1)


#### List Models ####

def get_model_for_conformal(model_name, list_quantiles_forest, params_qforest, **kwargs):
    if model_name == 'qrf':
        quantile_estimator = helper.QuantileForestRegressorAdapter(model=None,
                                                                   fit_params=None,
                                                                   quantiles=list_quantiles_forest,
                                                                   params=params_qforest)
    elif model_name == 'qgam':
        location_file = os.path.join(os.getcwd(), 'results\\predictions\\')
        location_file = os.path.join(location_file, kwargs['filename'])
        quantile_estimator = helper.CustomExternalRegressorAdapter(model=None,
                                                                   fit_params={},
                                                                   params={'location_file': location_file})

    return quantile_estimator


def get_gridsearch_models(model_name, features, params, quantiles=None):
    all_models = []
    keys, values = zip(*params.items())
    param_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for i, param_dict in enumerate(param_grid):
        if model_name == 'lasso':
            all_models.append(SklearnModelPipeline(f'Lasso_{i}', Pipeline([
                ('rgr', Lasso(max_iter=400000, alpha=param_dict['alpha'], solver ='highs'))
            ]), features))

        if model_name == 'xgblin':
            all_models.append(SklearnModelPipeline(f'XGBRegressor_linear_{i}', Pipeline([
                ('rgr', XGBRegressor(booster='gblinear',
                                     n_estimators=param_dict['n_estimators'],
                                     max_depth=param_dict['max_depth'],
                                     learning_rate=param_dict['learning_rate'],
                                     subsample=param_dict['subsample'],
                                     reg_alpha=param_dict['reg_alpha'],
                                     colsample_bytree=param_dict['colsample_bytree']
                                     ))
            ]), features))

        elif model_name == 'xgbtree':
            all_models.append(SklearnModelPipeline(f'XGBRegressor_tree_{i}', Pipeline([
                ('rgr', XGBRegressor(booster='gbtree',
                                     n_estimators=param_dict['n_estimators'],
                                     max_depth=param_dict['max_depth'],
                                     learning_rate=param_dict['learning_rate'],
                                     subsample=param_dict['subsample'],
                                     reg_alpha=param_dict['reg_alpha'],
                                     colsample_bytree=param_dict['colsample_bytree']
                                     ))
            ]), features))

        elif model_name == 'rf':
            all_models.append(SklearnModelPipeline(f'RandomForestRegressor_{i}', Pipeline([
                ('rgr', RandomForestRegressor(max_depth=param_dict['max_depth'],
                                              n_estimators=param_dict['n_estimators'],
                                              max_features='sqrt',
                                              random_state=0))
            ]), features))

        elif model_name == 'qrf':
            all_models.append(SklearnModelPipeline(f'QuantileRandomForestRegressor_{i}',
                                                   Pipeline([('rgr', RandomForestQuantileRegressor(
                                                       n_estimators=param_dict['n_estimators'],
                                                       max_features='sqrt',
                                                       max_depth=param_dict['max_depth'],
                                                       random_state=0
                                                   ))
                                                             ]), features))
        elif model_name == 'qlasso':
            for q in quantiles:
                all_models.append(SklearnModelPipeline(f'LassoQuantile{i}_{q}', Pipeline([
                    ('rgr', QuantileRegressor(alpha=param_dict["alpha"], quantile=q / 100, solver='highs'))
                ]), features))

        elif model_name == 'qgb':
            for q in quantiles:
                all_models.append(SklearnModelPipeline(f'GradientBoosting{i}_{q}', Pipeline([
                    ('rgr', GradientBoostingRegressor(loss='quantile', alpha=q / 100,
                                                      n_estimators=param_dict['n_estimators'],
                                                      max_features='sqrt',
                                                      learning_rate=param_dict['learning_rate'],
                                                      subsample=param_dict['subsample'],
                                                      max_depth=param_dict['max_depth']))
                ]), features))

    return all_models

def get_mean_models(features, models='all'):
    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {'alpha': np.logspace(np.log10(0.0005), np.log10(2), 20)}
    all_models = []

    for max_train_size in [ 3 * 30, 6 * 30, 9 * 30, 12 * 30, 24 * 30, 36 * 30, None]:
        if models == 'all' or models == 'lassobest':
            all_models.append(
                SklearnModelPipeline(
                    f'Lasso_alpha_1PER{max_train_size}',
                    Pipeline([
                        ('scale', StandardScaler()),
                        ('rgr', Lasso(max_iter=4000000, alpha=0.005)),
                    ]),
                    features, max_train_size=max_train_size))

        if models == "all" or models == 'lassocv':
            all_models.append(SklearnModelPipeline(f'LassoCVPER{max_train_size}', Pipeline([
                ('scale', StandardScaler()),
                ('rgr', LassoCV(cv=tscv, max_iter=4000000))
            ]), features, max_train_size=max_train_size))

        if models == 'all' or models == 'xgblin':
            all_models.append(SklearnModelPipeline(f'XGBRegressor_linear_PER{max_train_size}', Pipeline([
                ('scale', StandardScaler()),
                ('rgr', XGBRegressor(booster='gblinear',
                                     reg_alpha=0.0,
                                     max_depth=10,
                                     n_estimators=200,
                                     learning_rate=0.05,
                                     subsample=0.6,
                                     colsample_bytree=1.))
            ]), features, max_train_size=max_train_size))  # bEST : alpha = 1.15
        if models == "all" or models == 'xgbtree':
            all_models.append(SklearnModelPipeline(f'XGBRegressor_treesPER{max_train_size}', Pipeline([
                ('rgr', XGBRegressor(max_depth=4, subsample=0.5, colsample_bytree=0.33, alpha=2, n_estimators=25))
            ]), features, max_train_size=max_train_size))

        if models == 'all' or models == 'rf':
            all_models.append(SklearnModelPipeline(f'RandomForestRegressorPER{max_train_size}', Pipeline([
                ('rgr', RandomForestRegressor(max_depth=10, n_estimators=400, max_features='sqrt'))
            ]), features, max_train_size=max_train_size))

    return all_models

def get_quantile_models(features, quantiles, models='all'):
    quantile_models = []

    with open(os.path.join(DIRECTORY, "params/params_qrf.pkl"), "rb") as file:
        dic_var_qrf = pickle.load(file)
    if models == 'all' or models == 'qrf':
        for max_train_size in [3 * 30, 6 * 30, 9 * 30, 12 * 30, 24 * 30, 36 * 30, None]:
            quantile_models.append(SklearnModelPipeline(f'QuantileRandomForestRegressorPER{max_train_size}',
                        Pipeline([('rgr', RandomForestQuantileRegressor(random_state=0, max_depth=10, n_estimators=400))
                             ]), features, max_train_size=max_train_size))

    for q in quantiles:
        if 'simple_test' in models:
            quantile_models.append(SklearnModelPipeline(f'Lasso1QuantilePER{12*30}_{q}', Pipeline([
                    ('rgr', QuantileRegressor(alpha=0.005, quantile=q / 100, solver='highs'))
                ]), features, max_train_size=12*30))

        for max_train_size in [3 * 30, 6 * 30, 9 * 30, 12 * 30, 24 * 30, 36 * 30, None]:

            if models == 'all' or models == 'all_but_qrf' or models == 'lasso2':
                quantile_models.append(SklearnModelPipeline(f'Lasso2QuantilePER{max_train_size}_{q}', Pipeline([
                    ('rgr', QuantileRegressor(alpha=0.5, quantile=q / 100, solver='highs'))
                ]), features, max_train_size=max_train_size))

            if models == 'all' or models == 'all_but_qrf' or models == 'lasso1':
                quantile_models.append(SklearnModelPipeline(f'Lasso1QuantilePER{max_train_size}_{q}', Pipeline([
                    ('rgr', QuantileRegressor(alpha=0.005, quantile=q / 100, solver='highs'))
                ]), features, max_train_size=max_train_size))

            if models == 'all' or models == 'all_but_qrf' or models == 'linear':
                quantile_models.append(SklearnModelPipeline(f'LinQuantilePER{max_train_size}_{q}', Pipeline([
                    ('rgr', QuantileRegressor(alpha=0., quantile=q / 100, solver='highs'))
                ]), features, max_train_size=max_train_size))

            if models == 'all' or models == 'all_but_qrf' or  models == 'gb':
                quantile_models.append(SklearnModelPipeline(f'GradientBoostingPER{max_train_size}_{q}', Pipeline([
                    ('rgr', GradientBoostingRegressor(loss='quantile', alpha=q / 100,
                                                      n_estimators=200,
                                                      max_features='sqrt',
                                                      learning_rate=0.05,
                                                      subsample=0.6,
                                                      max_depth=4))
                ]), features, max_train_size=max_train_size))

    return quantile_models
