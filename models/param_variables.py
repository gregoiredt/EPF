import numpy as np

DIC_PARAM_VARIABLES = {
    'lasso' : {
        'alpha' : np.logspace(np.log10(0.000001), np.log10(1), 50)
    },
    'xgblin' : {
        'n_estimators' : [50, 100, 200],
        'max_depth' : [4, 7, 10],
        'learning_rate' : [0.005, 0.01, 0.05, 0.01],
        'subsample' : [1, 0.6],
        'reg_alpha' : [0., 0.1, 0.3],
        'colsample_bytree' : [1., 0.6]
    }, # 3 *3 * 4  * 2 * 2 * 3 = 438
    'xgbtree' : {
        'n_estimators' : [50, 100, 200],
        'max_depth' : [4, 7, 10],
        'learning_rate' : [0.005, 0.01, 0.05, 0.01],
        'subsample' : [1, 0.6],
        'reg_alpha' : [0., 0.1, 0.3],
        'colsample_bytree' : [1., 0.6]
    }, # 3 *3 * 4  * 2 * 2 * 3 = 438

    'rf' : {
        'max_depth' : [3, 4, 5, 7, 10],
        'n_estimators' : [50, 100, 150, 200, 400]
    }, # 5 * 5 = 25

    'qrf': {
        'max_depth': [3, 4, 5, 7, 10],
        'n_estimators': [50, 100, 150, 200, 400]
    },  # 5 * 5 = 25

    'qlasso' : {'alpha': np.logspace(np.log10(0.005), np.log10(2), 10)},

    'qgb' : {
        'n_estimators' : [50, 100, 200],
        'max_depth' : [4, 7, 10],
        'learning_rate' : [0.01, 0.05, 0.01],
        'subsample' : [1, 0.6]
    } # 3*3*3*2 = 54
}