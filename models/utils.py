from sklearn.model_selection import TimeSeriesSplit
import numpy as np
MAX_TRAIN_SIZE = 4 * 365

def custom_time_series_split(df, mode='week', cal=False, cal_size=0.5, **kwargs):
    '''

    :param cal_size:
    :param cal:
    :param df: DataFrame - Data to split between train and test
    :param mode: str - 'week' or 'day'
    :return:
    '''

    max_train_size = kwargs['max_train_size'] if 'max_train_size' in kwargs else MAX_TRAIN_SIZE

    years = sorted(df.Time.dt.year.unique()[1:])
    df["week"] = df.Time.dt.isocalendar().week

    for y in years:
        df.loc[df.Time.dt.year == y, 'week'] += df.loc[df.Time.dt.year == y - 1, 'week'].max()

    if mode == 'week':
        tscv = TimeSeriesSplit(max_train_size=max_train_size, n_splits=df.Date.nunique() // 7 - 1,
                               test_size=7)  # Premier jour est un lundi
        indexes = [split for i, split in enumerate(tscv.split(df)) if len(split[0]) == max_train_size]

    elif mode == 'day':
        tscv = TimeSeriesSplit(max_train_size=max_train_size, n_splits=df.Date.nunique() - 1, test_size=1)
        indexes = [split for i, split in enumerate(tscv.split(df)) if len(split[0]) == max_train_size]

    elif mode == 'deep':
        tscv = TimeSeriesSplit(max_train_size=max_train_size * 24,
                               n_splits=df.Date.nunique() // kwargs['prediction_length'] - 1,
                               test_size=kwargs['prediction_length'])
        indexes = [split for i, split in enumerate(tscv.split(df)) if len(split[0]) == max_train_size]

    if cal:
        indexes = [(split0[:int(len(split0) * (1 - cal_size))], split0[int(len(split0) * (1 - cal_size)):], split1)
                   for (split0, split1) in indexes]

    return indexes


def divide_in_equal_length(id_start, id_stop, n_div=10):
    if n_div >= id_stop - id_start:
        n_div = 1

    return [(id_start + int(i / n_div * (id_stop - id_start)), id_start + int((i +1) / n_div * (id_stop - id_start)))
            for i in range(n_div)]

def arcsin_transformation(x):
    return np.log(x + np.sqrt(np.square(x) + 1))

def inverse_arcsin_transformation(x):
    return  (np.exp(x) - np.exp(-x)) / 2