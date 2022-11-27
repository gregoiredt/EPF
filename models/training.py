import logging

import numpy as np
import pandas as pd
from pandas import DataFrame, concat
from time import time
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error
from numpy import mean, sqrt, zeros
from loguru import logger
from skgarden import RandomForestQuantileRegressor
import multiprocessing as mp
from .utils import custom_time_series_split, divide_in_equal_length
from conformal.cqr import helper
from conformal.nonconformist.nc import RegressorNc
from conformal.nonconformist.nc import QuantileRegErrFunc
from .metrics import mean_pinball_loss, unconditional_coverage, empirical_coverage, wrinkler_score, interval_width
from .models import AsinhMedianScaler, MedianScaler


def cqr_training(df, features, list_quantiles_forest, params_qforest, dic_var_qrf, significance_levels, mode,
                 list_cal_size, num_cpus, **kwargs):
    id_start, id_stop = kwargs['id_start'], kwargs['id_stop']
    preprocessing = kwargs['preprocessing'] if 'preprocessing' in kwargs else False
    list_hours = kwargs['list_hours']
    method = kwargs['method']
    dic_method_params = kwargs['method_params']
    no_conformal = kwargs['no_conformal']
    arguments = []
    list_divides = divide_in_equal_length(id_start, id_stop, n_div=kwargs['n_div'])
    parallel = kwargs['parallel'] if 'parallel' in kwargs else True

    logger.info(f'Parameters for submitted job...')

    logger.info(f'Params forest = {params_qforest}')
    logger.info(f'List quantile : {list_quantiles_forest}')
    logger.info(f'List coverages : {significance_levels}')
    logger.info(f'N_divides = {len(list_divides)}')
    logger.info(f'preprocessing = {preprocessing}')
    logger.info(f'mode = {mode}')
    logger.info(f'method = {method}')
    logger.info(f'cal_sizes = {list_cal_size}')
    logger.info(f'list_hours = {list_hours}')
    logger.info(f'id_start, id_stop = {(id_start, id_stop)}')
    logger.info(f'no_conformal = {no_conformal}')

    for method_param in dic_method_params[method]:
        for cal_size in list_cal_size:
            for h in list_hours:
                for (start, stop) in list_divides:
                    arguments.append((df, h, features, list_quantiles_forest, params_qforest, dic_var_qrf,
                                      significance_levels, mode, cal_size, method, method_param, start, stop,
                                      no_conformal, preprocessing))

    logger.info(f'Launching {len(arguments)} jobs on {num_cpus}...')

    if parallel:
        pool = mp.Pool(num_cpus)
        if kwargs['no_conformal']:
            logger.debug('Training test version of CQR')
            res = pool.starmap(_cqr_training_test_conformal, arguments)
        elif method in ['CQRoj', 'CQRoj+', 'ACI_CQRoj']:
            logger.debug('Training CQR with Jacknife framework')
            logger.debug(f'Number of runs : {len(arguments)}')
            res = pool.starmap(_cqr_training_jacknife, arguments)
        else:
            res = pool.starmap(_cqr_training, arguments)
        pool.close()
    else:
        res = []
        for i, arg in enumerate(arguments):
            if kwargs['no_conformal']:
                logger.debug('Training test version of CQR')
                res.append(_cqr_training_test_conformal(*arg, parallel=False))
            elif method in ['CQRoj', 'CQRoj+', 'ACI_CQRoj']:
                logger.debug('Training CQR with Jacknife framework')
                res.append(_cqr_training_jacknife(*arg, parallel=False))
            else:
                res.append(_cqr_training(*arg, parallel=False))

    return pd.concat(res)


def _update_params_qforest(params_qforest, dic_var, hour):
    for key, value in dic_var[hour].items():
        params_qforest[key] = value
    return params_qforest


def _cqr_training_test_conformal(df, h, features, list_quantiles_forest, params_qforest, dic_var_qrf,
                                 significance_levels, mode,
                                 cal_size, method, method_param, id_start, id_stop, no_conformal, preprocessing, parallel=True):
    df_pred = DataFrame()
    dfh = df[df.Hour == h].reset_index(drop=True)
    list_split = custom_time_series_split(dfh, mode=mode, cal=True, cal_size=cal_size)[id_start: id_stop]
    #params_qforest = _update_params_qforest(params_qforest, dic_var_qrf, h)

    gammas = method_param if method == 'ACI' else None
    adaptative_significance_levels = np.repeat(np.array(significance_levels).reshape(-1, 1), len(gammas), axis=1) \
        if method == 'ACI' else None

    list_quantiles = sorted([q[0] for q in list_quantiles_forest] + [q[1] for q in list_quantiles_forest])

    for i, (idx_train, idx_cal, idx_val) in enumerate(list_split):
        X_train, y_train = dfh.loc[:, features].to_numpy(), dfh.loc[:, 'SpotPrice'].to_numpy() # Indices will be selected later later
        X_cal, y_cal = dfh.loc[idx_cal, features].to_numpy(), dfh.loc[idx_cal, 'SpotPrice'].to_numpy()
        X_val, y_val = dfh.loc[idx_val, features].to_numpy(), dfh.loc[idx_val, 'SpotPrice'].to_numpy()

        quantile_estimator = helper.QuantileForestRegressorAdapter(model=None,
                                                                   fit_params=None,
                                                                   quantiles=list_quantiles_forest,
                                                                   params=params_qforest,
                                                                   preprocessing=preprocessing)
        nc = RegressorNc(quantile_estimator,
                         QuantileRegErrFunc(),
                         method='CQR'
                         )


        if method == 'CQRcal':
            predictions, predictions_no_conformal = helper.run_icp(nc=nc,
                                                                   X_train=X_train,
                                                                   y_train=y_train,
                                                                   X_test=X_cal,
                                                                   idx_train=idx_train,
                                                                   idx_cal=idx_cal,
                                                                   significance=significance_levels,
                                                                   y_test=y_cal,
                                                                   method='no_conformal')
            if no_conformal:

                predictions = predictions.reshape((-1, len(significance_levels) * 2))
                predictions_no_conformal = predictions_no_conformal.reshape((-1, len(significance_levels) * 2))

                for j, quantile in enumerate(list_quantiles):
                    model_name = f'cqrcCAL{int(100 * cal_size)}'
                    new_df_pred = pd.DataFrame({
                        'Hour': h,
                        'SpotPrice': y_cal,
                        'SpotPrice_pred': predictions[:, j],
                        'quantile': quantile,
                        'Date': dfh.loc[idx_cal, 'Date'],
                        'model': model_name
                    })

                    df_pred = df_pred.append(new_df_pred)

                    model_name = f'cqrcCAL{int(100 * cal_size)}TYPEnoconformal'
                    new_df_pred = pd.DataFrame({
                        'Hour': h,
                        'SpotPrice': y_cal,
                        'SpotPrice_pred': predictions_no_conformal[:, j],
                        'quantile': quantile,
                        'Date': dfh.loc[idx_cal, 'Date'],
                        'model': model_name
                    })

                    df_pred = df_pred.append(new_df_pred)

        if method == 'CQRval':
            predictions, predictions_no_conformal = helper.run_icp(nc, X_train, y_train, X_val, idx_train, idx_cal,
                                                                   significance_levels, y_test=y_val,
                                                                   method='no_conformal')

            if no_conformal:
                predictions = predictions.reshape((-1, len(significance_levels) * 2))
                predictions_no_conformal = predictions_no_conformal.reshape((-1, len(significance_levels) * 2))

                for j, quantile in enumerate(list_quantiles):
                    model_name = f'cqrvCAL{int(100 * cal_size)}'
                    new_df_pred = pd.DataFrame({
                        'Hour': h,
                        'SpotPrice': y_val,
                        'SpotPrice_pred': predictions[:, j],
                        'quantile': quantile,
                        'Date': dfh.loc[idx_val, 'Date'],
                        'model': model_name
                    })

                    df_pred = df_pred.append(new_df_pred)

                    model_name = f'cqrvCAL{int(100 * cal_size)}TYPEnoconformal'
                    new_df_pred = pd.DataFrame({
                        'Hour': h,
                        'SpotPrice': y_val,
                        'SpotPrice_pred': predictions_no_conformal[:, j],
                        'quantile': quantile,
                        'Date': dfh.loc[idx_val, 'Date'],
                        'model': model_name
                    })

                    df_pred = df_pred.append(new_df_pred)
    return df_pred


def _cqr_training_jacknife(df, h, features, list_quantiles_forest, params_qforest, dic_var_qrf, significance_levels,
                           mode, cal_size, method, method_param, id_start, id_stop, no_conformal, preprocessing, parallel=True):
    df_pred = DataFrame()
    dfh = df[df.Hour == h].reset_index(drop=True)
    (idx_train, idx_cal, idx_val) = custom_time_series_split(dfh, mode=mode, cal=True, cal_size=cal_size)[0]
    #params_qforest = _update_params_qforest(params_qforest, dic_var_qrf, h)

    # ACI possibility
    gammas = method_param if 'ACI' in method else None
    adaptative_significance_levels = np.repeat(np.array(significance_levels).reshape(-1, 1), len(gammas), axis=1) \
        if 'ACI' in method else None
    nc_method = 'ACI' if 'ACI' in method else 'CQR'
    method = 'CQRoj' if 'ACI' in method else method

    idx_train_cal = np.concatenate([idx_train, idx_cal])

    if len(list_quantiles_forest) > 1:
        list_quantiles = sorted([q[0] for q in list_quantiles_forest] + [q[1] for q in list_quantiles_forest])

    X_train, y_train = dfh.loc[idx_train_cal, features].to_numpy(), dfh.loc[idx_train_cal, 'SpotPrice'].to_numpy()
    id_start_val, id_stop_val = np.max(idx_cal) + 1 + id_start, np.max(idx_cal) + 1 + id_stop
    X_val, y_val = dfh.loc[id_start_val:id_stop_val, features].to_numpy(), dfh.loc[id_start_val:id_stop_val, 'SpotPrice'].to_numpy()

    quantile_estimator = helper.QuantileForestRegressorAdapter(model=None,
                                                               fit_params=None,
                                                               quantiles=list_quantiles_forest,
                                                               params=params_qforest,
                                                               preprocessing=preprocessing)


    nc = RegressorNc(quantile_estimator,
                     QuantileRegErrFunc(),
                     method=nc_method,
                     gammas=gammas,
                     significance_t=adaptative_significance_levels
                     )

    time_start = time()
    if method == 'CQRoj':
        predictions = helper.run_ojacknife(nc, X_train, y_train, X_val, idx_train, idx_cal, significance_levels,
                                           y_test=y_val.copy(), print_=not(parallel), method=nc_method)
    if method == 'CQRoj+':
        predictions = helper.run_ojacknifeplus(nc, X_train, y_train, X_val, idx_train, idx_cal, significance_levels,
                                               y_test=y_val.copy(), print_=not(parallel), method=nc_method)
    time_training = time() - time_start

    if nc_method=='CQR':

        predictions = predictions.reshape((-1, len(significance_levels) * 2))
        predictions = np.sort(predictions, axis=1)

        for j, quantile in enumerate(list_quantiles):
            model_name = f'{method}CAL{int(100 * cal_size)}'
            new_df_pred = pd.DataFrame({
                'Hour': h,
                'SpotPrice': y_val,
                'SpotPrice_pred': predictions[:, j],
                'quantile': quantile,
                'Date': dfh.loc[id_start_val:id_stop_val, 'Date'],
                'model': model_name,
                'training_time': time_training / (len(list_quantiles) * predictions.shape[0])
            })
            df_pred = df_pred.append(new_df_pred)

    elif nc_method == 'ACI':
        predictions = predictions.reshape((-1, len(significance_levels) * 2, len(gammas)))
        predictions = np.sort(predictions, axis=1)

        for idg, gamma in enumerate(gammas):
            for j, quantile in enumerate(list_quantiles):
                model_name = f'aciGAMMA{gamma}CAL{int(100 * cal_size)}'
                new_df_pred = pd.DataFrame({
                    'Hour': h,
                    'SpotPrice': y_val,
                    'SpotPrice_pred': predictions[:, j, idg],
                    'quantile': quantile,
                    'Date': dfh.loc[id_start_val:id_stop_val, 'Date'],
                    'model': model_name,
                    'training_time': time_training / (len(list_quantiles) * predictions.shape[0])
                })
                df_pred = df_pred.append(new_df_pred)

    return df_pred


def _cqr_training(df, h, features, list_quantiles_forest, params_qforest, dic_var_qrf, significance_levels, mode,
                  cal_size, method, method_param, id_start, id_stop, no_conformal, preprocessing, parallel=True):
    df_pred = DataFrame()
    dfh = df[df.Hour == h].reset_index(drop=True)
    list_split = custom_time_series_split(dfh, mode=mode, cal=True, cal_size=cal_size)[id_start: id_stop]
    #params_qforest = _update_params_qforest(params_qforest, dic_var_qrf, h)

    gammas = method_param if method == 'ACI' else None
    adaptative_significance_levels = np.repeat(np.array(significance_levels).reshape(-1, 1), len(gammas), axis=1) \
        if method == 'ACI' else None

    if len(list_quantiles_forest) > 1:
        list_quantiles = sorted([q[0] for q in list_quantiles_forest] + [q[1] for q in list_quantiles_forest])

    adaptative_quantiles = np.array(list_quantiles_forest) if method == 'aCQR' else None

    for i, (idx_train, idx_cal, idx_val) in enumerate(list_split):
        X_train, y_train = dfh.loc[:, features].to_numpy(), dfh.loc[:,
                                                            'SpotPrice'].to_numpy()  # We give index after that, no need to exclude idx_val
        X_val, y_val = dfh.loc[idx_val, features].to_numpy(), dfh.loc[idx_val, 'SpotPrice'].to_numpy()

        if method in ['CQR', 'ACI']:

            quantile_estimator = helper.QuantileForestRegressorAdapter(model=None,
                                                                       fit_params=None,
                                                                       quantiles=list_quantiles_forest,
                                                                       params=params_qforest)
        elif method == 'aCQR':
            assert mode == 'day', 'Adaptative QRF only works for daily update'
            quantile_estimator = helper.AdaptativeQuantileForestRegressorAdapter(model=None,
                                                                                 gamma=method_param,
                                                                                 quantiles=list_quantiles_forest,
                                                                                 adaptative_quantiles=adaptative_quantiles,
                                                                                 fit_params=None,
                                                                                 params=params_qforest,
                                                                                 preprocessing=preprocessing)
        nc = RegressorNc(quantile_estimator,
                         QuantileRegErrFunc(),
                         method=method,
                         gammas=gammas,
                         significance_t=adaptative_significance_levels
                         )
        time_start = time()
        predictions = helper.run_icp(nc, X_train, y_train, X_val, idx_train, idx_cal, significance_levels,
                                     y_test=y_val)

        time_training = time() - time_start

        if method == 'aCQR':
            model_name = f'cqrCAL{int(100 * cal_size)}GAMMA{method_param}'
            predictions = predictions.reshape((-1, len(significance_levels) * 2))
            predictions = np.sort(predictions, axis=1)

            list_aquantiles = sorted([q[0] for q in adaptative_quantiles] + [q[1] for q in adaptative_quantiles])

            for j, quantile in enumerate(list_quantiles):
                new_df_pred = pd.DataFrame({
                    'Hour': h,
                    'SpotPrice': y_val,
                    'SpotPrice_pred': predictions[:, j],
                    'quantile': quantile,
                    'adaptative_quantile': list_aquantiles[j],
                    'Date': dfh.loc[idx_val, 'Date'],
                    'model': model_name,
                    'training_time': time_training / (len(list_quantiles) * predictions.shape[0])
                })
                df_pred = df_pred.append(new_df_pred)

            adaptative_quantiles = quantile_estimator.update_quantiles(y_val.reshape(-1),
                                                                       predictions[0].reshape(-1))

        if method == 'CQR':

            predictions = predictions.reshape((-1, len(significance_levels) * 2))
            predictions = np.sort(predictions, axis=1)

            for j, quantile in enumerate(list_quantiles):
                model_name = f'cqrCAL{int(100 * cal_size)}'
                new_df_pred = pd.DataFrame({
                    'Hour': h,
                    'SpotPrice': y_val,
                    'SpotPrice_pred': predictions[:, j],
                    'quantile': quantile,
                    'Date': dfh.loc[idx_val, 'Date'],
                    'model': model_name,
                    'training_time': time_training / (len(list_quantiles) * predictions.shape[0])
                })

                df_pred = df_pred.append(new_df_pred)

        if method == 'ACI':
            # Updating significance_levels
            adaptative_significance_levels = nc.significance_t
            predictions = predictions.reshape((-1, len(significance_levels) * 2, len(gammas)))

            for idg, gamma in enumerate(gammas):
                for j, quantile in enumerate(list_quantiles):
                    model_name = f'aciGAMMA{gamma}CAL{int(100 * cal_size)}'
                    new_df_pred = pd.DataFrame({
                        'Hour': h,
                        'SpotPrice': y_val,
                        'SpotPrice_pred': predictions[:, j, idg],
                        'quantile': quantile,
                        'Date': dfh.loc[idx_val, 'Date'],
                        'model': model_name,
                        'training_time': time_training / (len(list_quantiles) * predictions.shape[0])
                    })
                    df_pred = df_pred.append(new_df_pred)

    return df_pred


def _sklearn_quantile_training(skmodel, df, features, quantiles, hour, mode, cal, cal_size, id_start, id_stop,
                               max_train_size, preprocessing, gridsearch):
    model_name = skmodel.name
    is_qrf = isinstance(skmodel.get_model(0).named_steps['rgr'], RandomForestQuantileRegressor)
    df_pred = DataFrame()

    dfh = df[(df.Hour == hour)].reset_index(drop=True)
    if not gridsearch:
        indexes_split = custom_time_series_split(dfh, mode, cal, cal_size, max_train_size=max_train_size)[id_start: id_stop]
    else:
        indexes_split = custom_time_series_split(dfh, mode, max_train_size=3*365)[0:350:10]

    if not cal:
        for i, (train_index, val_index) in enumerate(indexes_split):
            err_train = False
            X_train, y_train = dfh.loc[train_index, features], dfh.loc[train_index, 'SpotPrice']
            X_val, y_val = dfh.loc[val_index, features], dfh.loc[val_index, 'SpotPrice']

            x_scaler = MedianScaler()
            x_scaler.fit(X_train)
            X_train, X_val = x_scaler.transform(X_train), x_scaler.transform(X_val)

            if preprocessing:
                y_scaler = AsinhMedianScaler()
                y_scaler.fit(y_train)
                y_train = y_scaler.transform(y_train)

            time_start = time()
            try:
                skmodel.train(X_train, y_train, hour)
            except TypeError:
                err_train = True

            time_training = time() - time_start
            if not is_qrf:
                y_pred = skmodel.predict(X_val, hour) if not err_train else np.array([np.nan for i in range(len(y_val))])
                if preprocessing:
                    y_pred = y_scaler.inverse_transform(y_pred)

                new_df_pred = pd.DataFrame({
                    'Hour': dfh.Hour.unique()[0],
                    'SpotPrice': y_val,
                    'SpotPrice_pred': y_pred,
                    'quantile': float(model_name.split('_')[-1]),
                    'model': model_name.split('_')[0],
                    'Date': dfh.loc[val_index, 'Date'],
                    'training_time': time_training / (len(y_pred))
                })
                df_pred = df_pred.append(new_df_pred)
            else:
                for quantile in quantiles:
                    y_pred = skmodel.predict(X_val, hour, quantile=quantile)
                    if preprocessing:
                        y_pred = y_scaler.inverse_transform(y_pred)

                    new_df_pred = pd.DataFrame({
                        'Hour': dfh.Hour.unique()[0],
                        'SpotPrice': y_val,
                        'SpotPrice_pred': y_pred,
                        'model': model_name,
                        'quantile': quantile,
                        'Date': dfh.loc[val_index, 'Date'],
                        'training_time': time_training / (len(y_pred) * len(quantiles))
                    })
                    df_pred = df_pred.append(new_df_pred)
        return df_pred

    if cal:
        raise ValueError('Not implemented yet')


def sklearn_training(skl_models, df, features, quantiles=None, num_cpus=1, **kwargs):
    gridsearch = kwargs['gridsearch'] if 'gridsearch' in kwargs else False
    preprocessing = kwargs['preprocessing'] if 'preprocessing' in kwargs else False
    mode = kwargs['time_mode'] if 'time_mode' in kwargs else 'day'
    cal = kwargs['cal'] if 'cal' in kwargs else False
    cal_size = kwargs['cal_size'] if 'cal_size' in kwargs else 0.5
    list_hours = kwargs['list_hours']
    id_start, id_stop = kwargs['id_start'], kwargs['id_stop']
    max_train_size = kwargs['max_train_size'] * 365
    feature_importance_param = kwargs['feature_importance_param'] if 'feature_importance_param' in kwargs else False
    list_divides = divide_in_equal_length(id_start, id_stop, n_div=kwargs['n_div'])

    logger.info(f'Parameters for submitted job...')
    logger.info(f'Num_models = {len(skl_models)}')
    logger.info(f'Quantiles = {quantiles}')
    logger.info(f'N_divides = {len(list_divides)}')
    logger.info(f'Gridsearch = {gridsearch}')
    logger.info(f'preprocessing = {preprocessing}')
    logger.info(f'mode = {mode}')
    logger.info(f'cal = {cal}')
    logger.info(f'cal_size = {cal_size}')
    logger.info(f'list_hours = {list_hours}')
    logger.info(f'id_start, id_stop = {(id_start, id_stop)}')
    logger.info(f'max_train_size = {max_train_size}')

    if feature_importance_param:
        logger.info('Going to compute feature importances...')

    df_res = DataFrame()

    arguments = []
    if not (quantiles is None):  # Quantile models
        for (start, stop) in list_divides:
            for quantile_model in skl_models:
                for h in list_hours:
                    arguments.append(
                        (quantile_model, df, features, quantiles, h, mode, cal, cal_size, start, stop, max_train_size, preprocessing, gridsearch))

        logger.info(f'LAUCHING ({len(arguments)} jobs on {num_cpus} cpus)')
        pool = mp.Pool(num_cpus)
        results = pool.starmap(_sklearn_quantile_training, arguments)

    else:  # Mean models
        for (start, stop) in list_divides:
            for model in skl_models:
                for h in list_hours:
                    arguments.append(
                        (model, df, features, h, mode, start, stop, max_train_size, feature_importance_param, preprocessing, gridsearch))

        logger.info(f'LAUCHING Gridsearch ({len(arguments)} jobs on {num_cpus} cpus)')
        pool = mp.Pool(num_cpus)
        results = pool.starmap(_sklearn_training, arguments)
    pool.close()

    df_pred = concat(results).sort_values(by=['Date', 'Hour'])
    return df_pred


def _sklearn_training(skmodel, df, features, hour, mode, start, stop, max_train_size, feature_importance_param, preprocessing, gridsearch):
    model_name = skmodel.name
    df_pred = DataFrame()
    dfh = df[(df.Hour == hour)].reset_index(drop=True)

    if not gridsearch:
        indexes_split = custom_time_series_split(dfh, mode, max_train_size=max_train_size)[start:stop]
    else:
        indexes_split = custom_time_series_split(dfh, mode, max_train_size=3*365)[0:350:5] # 0:350:10

    for i, (train_index, val_index) in enumerate(indexes_split):
        err_train = False
        X_train, y_train = dfh.loc[train_index, features], dfh.loc[train_index, 'SpotPrice']
        X_val, y_val = dfh.loc[val_index, features], dfh.loc[val_index, 'SpotPrice']

        x_scaler = MedianScaler()
        x_scaler.fit(X_train)
        X_train, X_val = x_scaler.transform(X_train), x_scaler.transform(X_val)
        
        if preprocessing:
            y_scaler = AsinhMedianScaler()
            y_scaler.fit(y_train)
            y_train = y_scaler.transform(y_train)

        time_start = time()
        try:
            skmodel.train(X_train, y_train, hour)
        except TypeError:
            err_train = True

        time_training = time() - time_start

        if not feature_importance_param:
            if not err_train:
                y_pred = skmodel.predict(X_val, hour)
            else:
                y_pred = np.array([np.nan for i in len(y_val)])
            if preprocessing:
                y_pred = y_scaler.inverse_transform(y_pred)
            new_df_pred = pd.DataFrame({
                'Hour': dfh.Hour.unique()[0],
                'SpotPrice': y_val,
                'SpotPrice_pred': y_pred,
                'model': model_name,
                'Date': dfh.loc[val_index, 'Date'],
                'training_time': time_training / len(y_pred)
            })
            df_pred = df_pred.append(new_df_pred)

        else:
            method_fi = 'coef' if 'Lasso' in skmodel.name else 'MDI'
            df_fi = skmodel.get_feature_importance(hour, method=method_fi)
            df_fi['model'] = model_name
            df_fi['Date'] = dfh.iloc[val_index].Date.unique()[0]
            df_fi['Hour'] = dfh.Hour.unique()[0]
            df_pred = df_pred.append(df_fi)

    return df_pred

