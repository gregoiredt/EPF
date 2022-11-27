import pandas as pd
import pickle
from models.models import get_quantile_models, get_mean_models, get_gridsearch_models
from models.training import sklearn_training, cqr_training
from models.param_variables import DIC_PARAM_VARIABLES
import numpy as np
import argparse
import os
from preprocessing.preprocessing import process_and_save_data
from sklearn.preprocessing import OneHotEncoder
from loguru import logger

DIRECTORY = os.getcwd()  # os.path.join(os.getcwd(), LOCAL_DIRECTORY)
QUANTILES = np.array([0.5, 1, 2.5, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 97.5, 99, 99.5])
FEWER_QUANTILES = np.array([ 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99])
TAB_GAMMA = [0, 0.000005, 0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
             0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07,
             0.08, 0.09]


def load_data(filename_extension, overwrite=False, undesired_features=None):

    # Loading data
    try :
        if overwrite:
            logger.info('Erasing original file...')
            raise FileNotFoundError

        df = pd.read_csv(
            os.path.join(DIRECTORY,
                         f"data/data_cleaned_{filename_extension}.csv"))
    except FileNotFoundError as _:
        logger.info('File has not been found, Processing and saving data...')
        process_and_save_data(filename_extension)
        df = pd.read_csv(
            os.path.join(DIRECTORY,
                         f"data/data_cleaned_{filename_extension}.csv"))

    else:
        logger.info('Loading file...')
    a_file = open(os.path.join(DIRECTORY, f"data/dic_var_{filename_extension}.pkl"),
                  "rb")
    dic_var = pickle.load(a_file)
    a_file.close()

    # Handling times
    df['Time'] = pd.to_datetime(df.Date + "-" + df.Hour.astype(str), format="%Y-%m-%d-%H")
    logger.info(f"Date of Nans : {df[df.Lag_J7_Fossil_Hard_Coal.isna()].Date.unique()}")
    df = df[df.Date > '2016-01-10']

    # Dropping columns
    dic_keys_to_drop = [k for k in dic_var.keys() if ('autoregressive' in k) or ('invalid' in k)]
    columns_to_drop = []
    for key in dic_keys_to_drop:
        columns_to_drop += dic_var[key]
    columns_to_drop = list(set(columns_to_drop)
                           - {'SpotPrice'})
    df.drop(columns=columns_to_drop, inplace=True)

    if not (undesired_features is None):
        df.drop(columns=undesired_features, inplace=True)

    features = df.drop(columns=['Hour', 'Time', 'Date', 'SpotPrice']).columns

    return df, features


def gridsearch_pipeline(filename='gridsearch.csv', num_cpus =1, **kwargs):
    df, features = load_data(filename_extension='2016_2021')

    # One Hot Encoding
    df['weekday'] = df['weekday'].isin([5, 6]).astype(int)
    quantiles = QUANTILES if not kwargs['fewer'] else FEWER_QUANTILES

    all_models = get_gridsearch_models(model_name=kwargs['models'],
                                  features=features,
                                  params=DIC_PARAM_VARIABLES[kwargs['models']],
                                  quantiles=quantiles
                                  )
    if kwargs['models'] in ['lasso', 'xgblin', 'xgbtree', 'rf']:
        df_res = sklearn_training(all_models, df, features, num_cpus=num_cpus, gridsearch=True, **kwargs)
        logger.debug(f"Saving results to {'results/mreg/' + filename}...")

        df_res.to_csv('results/mreg/' + filename, index=False)

    elif kwargs["models"] in ['qlasso', 'qgb', 'qrf']:
        df_res = sklearn_training(all_models, df, features,
                                  quantiles=quantiles, num_cpus=num_cpus, gridsearch=True, **kwargs)
        logger.debug(f'Saving results to {"results/qreg/" + filename}...')
        df_res.to_csv('results/qreg/' + filename, index=False)


def mean_regression_pipeline(filename='mean_reg.csv', num_cpus=1, **kwargs):
    logger.debug('Loading data')

    df, features = load_data(filename_extension='2016_2021')

    # One Hot Encoding
    df['weekday'] = df['weekday'].isin([5, 6]).astype(int)

    # Load models
    mean_models = get_mean_models(features, models=kwargs['models'])
    logger.debug(f'Features : {features}')

    df_res = sklearn_training(mean_models, df, features, num_cpus=num_cpus, **kwargs)
    logger.debug(f"Saving results to {'results/mreg/' + filename}...")
    df_res.to_csv('results/mreg/' + filename, index=False)


def quantile_regression_pipeline(filename='quantile_reg.csv', num_cpus=1, **kwargs):
    logger.debug('Loading data...')
    df, features = load_data(filename_extension='2016_2021')

    # One Hot Encoding
    df['weekday'] = df['weekday'].isin([5, 6]).astype(int)

    quantile_models = get_quantile_models(features, QUANTILES, models=kwargs['models'])

    df_res = sklearn_training(quantile_models, df, features,
                              quantiles=QUANTILES, num_cpus=num_cpus, **kwargs)
    logger.debug(f'Saving results to {"results/qreg/" + filename}...')
    df_res.to_csv('results/qreg/' + filename, index=False)


def cqr_prediction_pipeline(filename='cqr_pred.csv', num_cpus=1, **kwargs):
    logger.debug('Loading data...')
    df, features = load_data(filename_extension='2016_2021')

    # One Hot Encoding
    df['weekday'] = df['weekday'].isin([5, 6]).astype(int)

    with open("params/params_qrf.pkl", "rb") as file:
        dic_var_qrf = pickle.load(file)

    # Init parameters
    significance_levels = np.array([1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]) / 100
    quantiles = [[q * 0.01, 1 - (q * 0.01)] for q in [0.5, 1, 2.5, 5, 10, 15, 20, 25, 30, 35, 40, 45]]
    list_quantiles_forest = [[round(q[0] * 100, 1), round(q[1] * 100, 1)] for q in quantiles]  \
        if kwargs['quantiles'] == 'all' else [0.5]

    params_qforest = dict()
    params_qforest["CV"] = kwargs['cqr_cv']
    params_qforest["coverage_factor"] = 0.85
    params_qforest["test_ratio"] = 0.05
    params_qforest["random_state"] = 1
    params_qforest["range_vals"] = 30
    params_qforest["num_vals"] = 10
    params_qforest["max_depth"] = 10
    params_qforest["n_estimators"] = 400
    params_qforest["max_features"] = 'sqrt'
    params_qforest["min_samples_leaf"] = 1

    kwargs['method_params'] = {
        'CQR': [None],
        'CQRcal': [None],
        'CQRval': [None],
        'aCQR' : np.array([0, 1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3, 2e-3, 5e-3, 7e-3, 1e-2, 5e-2]),
        'ACI': [TAB_GAMMA],
        'CQRoj' : [None],
        'CQRoj+' : [None],
        'ACI_CQRoj': [TAB_GAMMA],
    }

    mode = kwargs['time_mode']
    list_cal_size = kwargs['cal_sizes'] #[0.25, 0.5, 0.75]

    logger.debug('Launching Training...')
    df_res = cqr_training(df, features, list_quantiles_forest, params_qforest, dic_var_qrf, significance_levels, mode,
                          list_cal_size, num_cpus, **kwargs)
    logger.debug(f'Saving results to {"results/cqr/" + filename}')
    df_res.to_csv('results/cqr/' + filename, index=False)

#### Utils #####

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_cpus', type=int, required=False, default=1, help='Number of CPU to use')
    parser.add_argument('--type_training', type=str, required=True,
                        help='Type of training to execute (in ["conformal", "qreg", "mreg"])')
    parser.add_argument('--filename', type=str, required=False, default="pred.csv",
                        help='Filename to save, must end by ".csv"')
    parser.add_argument('--time_mode', type=str, required=False, default="day",
                        help='Window size of prediction, either day or week')
    parser.add_argument('--hours', type=str, required=False, default="all",
                        help='Hours to predict, either one hour (like 18) or all')
    parser.add_argument('--method', type=str, required=False, default='CQR',
                        help='Conformal Prediction method (either "CQR", "ACI")')
    parser.add_argument('--models', type=str, required=False, default='all',
                        help='Type of quantile model to train, either "all", "lasso", "gb", "qrf"')
    parser.add_argument('--id_start', type=int, required=False, default=0,
                        help='Time id from which we start predicting')
    parser.add_argument('--id_stop', type=int, required=False, default=730,
                        help='Time id from which we stop predicting')
    parser.add_argument('--n_div', type=int, required=False, default=1,
                        help='Number of time divide slices')
    parser.add_argument('--no_conformal', type=int, required=False, default=0,
                        help='Also perform a non conformalized quantile regression or not')
    parser.add_argument('--quantiles', type=str, required=False, default='all',
                        help='Whether to copute all quantiles or only the median')
    parser.add_argument('--max_train_size', type=int, required=False, default=4,
                        help='Maximum number of years on which to train')
    parser.add_argument('--feature_importance', type=int, required=False, default=0,
                        help='Whether to get feature_importances or not')
    parser.add_argument('--preprocessing', type=int, required=False, default=1,
                        help='Whether to preprocess data or not')
    parser.add_argument('--gridsearch', type=int, required=False, default=0, help='Special Gridsearch pipeline')
    parser.add_argument('--cal_sizes', type=str, required=False, default='all', help='cal_sizes to select')
    parser.add_argument('--fewer', type=str, required=False, default=0, help='Choose a limited amount of quantiles')
    parser.add_argument('--cqr_cv', type=int, required=False, default=0, help='Perform CV to choose coverage of QRF for CQR')
    parser.add_argument('--parallel', type=int, required=False, default=1, help='Choose parallel computation')
    return parser


def run_main():
    parser = get_parser()
    args = parser.parse_args()
    time_mode = args.time_mode
    gridsearch = bool(args.gridsearch)
    fewer = bool(args.fewer)
    no_conformal = True if args.no_conformal == 1 else 0
    compute_feature_importance = bool(args.feature_importance)
    preprocessing = bool(args.preprocessing)
    cal_sizes = [0.75, 0.5, 0.25] if args.cal_sizes == 'all' else [float(args.cal_sizes)]
    cqr_cv = bool(args.cqr_cv)
    parallel = bool(args.parallel)

    if len(args.hours) <= 2:
        list_hours = [int(args.hours)]
    elif args.hours == 'final_choice':
        list_hours = [3, 8, 13, 18, 23]
    else:
        list_hours = [i for i in range(24)]

    models = args.models

    if gridsearch:
        logger.debug('Starting GridSearch Pipeline...')
        gridsearch_pipeline(
            filename=args.filename,
            num_cpus=args.num_cpus,
            time_mode=time_mode,
            list_hours=list_hours,
            models=models,
            id_start=args.id_start,
            id_stop=args.id_stop,
            n_div=args.n_div,
            max_train_size=args.max_train_size,
            preprocessing=preprocessing,
            fewer=fewer
        )

    elif args.type_training == 'conformal':
        if (args.n_div > 1) and (args.method == 'ACI'):
            raise ValueError('Cannot perform ACI on a division. Are you sure you want to do this ?')

        logger.debug('Started conformal quantile regression training...')
        cqr_prediction_pipeline(filename=args.filename,
                                num_cpus=args.num_cpus,
                                time_mode=time_mode,
                                list_hours=list_hours,
                                method=args.method,
                                id_start=args.id_start,
                                id_stop=args.id_stop,
                                n_div=args.n_div,
                                no_conformal=no_conformal,
                                quantiles=args.quantiles,
                                max_train_size=args.max_train_size,
                                preprocessing=preprocessing,
                                cal_sizes=cal_sizes,
                                cqr_cv=cqr_cv,
                                parallel=parallel
                                )

    elif args.type_training == 'qreg':
        logger.debug('Started quantile regression training...')
        quantile_regression_pipeline(filename=args.filename,
                                     num_cpus=args.num_cpus,
                                     time_mode=time_mode,
                                     list_hours=list_hours,
                                     models=models,
                                     id_start=args.id_start,
                                     id_stop=args.id_stop,
                                     n_div=args.n_div,
                                     max_train_size=args.max_train_size,
                                     preprocessing=preprocessing
                                     )
    elif args.type_training == 'mreg':
        logger.debug('Started mean regression training...')

        mean_regression_pipeline(filename=args.filename,
                                 num_cpus=args.num_cpus,
                                 time_mode=time_mode,
                                 list_hours=list_hours,
                                 id_start=args.id_start,
                                 id_stop=args.id_stop,
                                 n_div=args.n_div,
                                 models=models,
                                 max_train_size=args.max_train_size,
                                 feature_importance_param=compute_feature_importance,
                                 preprocessing=preprocessing
                                 )

    logger.debug('Training finished !')


if __name__ == '__main__':
    run_main()

### Brouillon ###
