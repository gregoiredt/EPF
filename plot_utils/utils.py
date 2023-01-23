from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import TransformerMixin, _OneToOneFeatureMixin, BaseEstimator, clone
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from numpy import average, zeros, mean, log
from scipy.stats import chi2, norm 

MAX_TRAIN_SIZE = 4 * 365
DIC = {
    "Lag_J1_SpotPrice_H0": 'SpotPrice J-1',
    "Lag_J1_SpotPrice_H1": 'SpotPrice J-1',
    "Lag_J1_SpotPrice_H10": 'SpotPrice J-1',
    "Lag_J1_SpotPrice_H11": 'SpotPrice J-1',
    "Lag_J1_SpotPrice_H12": 'SpotPrice J-1',
    "Lag_J1_SpotPrice_H13": 'SpotPrice J-1',
    "Lag_J1_SpotPrice_H14": 'SpotPrice J-1',
    "Lag_J1_SpotPrice_H15": 'SpotPrice J-1',
    "Lag_J1_SpotPrice_H16": 'SpotPrice J-1',
    "Lag_J1_SpotPrice_H17": 'SpotPrice J-1',
    "Lag_J1_SpotPrice_H18": 'SpotPrice J-1',
    "Lag_J1_SpotPrice_H19": 'SpotPrice J-1',
    "Lag_J1_SpotPrice_H2": 'SpotPrice J-1',
    "Lag_J1_SpotPrice_H20": 'SpotPrice J-1',
    "Lag_J1_SpotPrice_H21": 'SpotPrice J-1',
    "Lag_J1_SpotPrice_H22": 'SpotPrice J-1',
    "Lag_J1_SpotPrice_H23": 'SpotPrice J-1',
    "Lag_J1_SpotPrice_H3": 'SpotPrice J-1',
    "Lag_J1_SpotPrice_H4": 'SpotPrice J-1',
    "Lag_J1_SpotPrice_H5": 'SpotPrice J-1',
    "Lag_J1_SpotPrice_H6": 'SpotPrice J-1',
    "Lag_J1_SpotPrice_H7": 'SpotPrice J-1',
    "Lag_J1_SpotPrice_H8": 'SpotPrice J-1',
    "Lag_J1_SpotPrice_H9": 'SpotPrice J-1',
    "Lag_J7_SpotPrice_H0": "SpotPrice J-7",
    "Lag_J7_SpotPrice_H1": "SpotPrice J-7",
    "Lag_J7_SpotPrice_H10": "SpotPrice J-7",
    "Lag_J7_SpotPrice_H11": "SpotPrice J-7",
    "Lag_J7_SpotPrice_H12": "SpotPrice J-7",
    "Lag_J7_SpotPrice_H13": "SpotPrice J-7",
    "Lag_J7_SpotPrice_H14": "SpotPrice J-7",
    "Lag_J7_SpotPrice_H15": "SpotPrice J-7",
    "Lag_J7_SpotPrice_H16": "SpotPrice J-7",
    "Lag_J7_SpotPrice_H17": "SpotPrice J-7",
    "Lag_J7_SpotPrice_H18": "SpotPrice J-7",
    "Lag_J7_SpotPrice_H19": "SpotPrice J-7",
    "Lag_J7_SpotPrice_H2": "SpotPrice J-7",
    "Lag_J7_SpotPrice_H20": "SpotPrice J-7",
    "Lag_J7_SpotPrice_H21": "SpotPrice J-7",
    "Lag_J7_SpotPrice_H22": "SpotPrice J-7",
    "Lag_J7_SpotPrice_H23": "SpotPrice J-7",
    "Lag_J7_SpotPrice_H3": "SpotPrice J-7",
    "Lag_J7_SpotPrice_H4": "SpotPrice J-7",
    "Lag_J7_SpotPrice_H5": "SpotPrice J-7",
    "Lag_J7_SpotPrice_H6": "SpotPrice J-7",
    "Lag_J7_SpotPrice_H7": "SpotPrice J-7",
    "Lag_J7_SpotPrice_H8": "SpotPrice J-7",
    "Lag_J7_SpotPrice_H9": "SpotPrice J-7",
    "Prev_Residual_Load_H0": "Prev_Residual_Load",
    "Prev_Residual_Load_H1": "Prev_Residual_Load",
    "Prev_Residual_Load_H10": "Prev_Residual_Load",
    "Prev_Residual_Load_H11": "Prev_Residual_Load",
    "Prev_Residual_Load_H12": "Prev_Residual_Load",
    "Prev_Residual_Load_H13": "Prev_Residual_Load",
    "Prev_Residual_Load_H14": "Prev_Residual_Load",
    "Prev_Residual_Load_H15": "Prev_Residual_Load",
    "Prev_Residual_Load_H16": "Prev_Residual_Load",
    "Prev_Residual_Load_H17": "Prev_Residual_Load",
    "Prev_Residual_Load_H18": "Prev_Residual_Load",
    "Prev_Residual_Load_H19": "Prev_Residual_Load",
    "Prev_Residual_Load_H2": "Prev_Residual_Load",
    "Prev_Residual_Load_H20": "Prev_Residual_Load",
    "Prev_Residual_Load_H21": "Prev_Residual_Load",
    "Prev_Residual_Load_H22": "Prev_Residual_Load",
    "Prev_Residual_Load_H23": "Prev_Residual_Load",
    "Prev_Residual_Load_H3": "Prev_Residual_Load",
    "Prev_Residual_Load_H4": "Prev_Residual_Load",
    "Prev_Residual_Load_H5": "Prev_Residual_Load",
    "Prev_Residual_Load_H6": "Prev_Residual_Load",
    "Prev_Residual_Load_H7": "Prev_Residual_Load",
    "Prev_Residual_Load_H8": "Prev_Residual_Load",
    "Prev_Residual_Load_H9": "Prev_Residual_Load",
    "Lag_J2_Fossil_Gas": "Production J-2",
    "Lag_J2_Fossil_Hard_Coal": "Production J-2",
    "Lag_J2_Fossil_Oil": "Production J-2",
    "Lag_J2_Hydro_Pumped_Storage": "Production J-2",
    "Lag_J2_Hydro_Water_Reservoir": "Production J-2",
    "Lag_J2_Nuclear": "Production J-2",
    "Lag_J7_Exchange_FR_DE": "Exchange",
    "Lag_J7_Exchange_FR_TOT": "Exchange",
    "Lag_J7_Fossil_Gas": "Production J-7",
    "Lag_J7_Fossil_Hard_Coal": "Production J-7",
    "Lag_J7_Fossil_Oil": "Production J-7",
    "Lag_J7_GazPrice": "Commodity price",
    "Lag_J1_GazPrice": "Commodity price",
    "M1_Coal": "Commodity price",
    "M1_Oil": "Commodity price",
    "Lag_J7_Hydro_Pumped_Storage": "Production J-7",
    "Lag_J7_Hydro_Water_Reservoir": "Production J-7",
    "Lag_J7_Nuclear": "Production J-7",
    "Nuclear_availability": "Nuclear_availability",
    "PublicHoliday_FR": 'Time feature',
    "SchoolHoliday_FR": 'Time feature',
    "clock": 'Time feature',
    "toy_cos": 'Time feature',
    "toy_sin": 'Time feature',
    "weekday": 'Time feature',
    "Ponts_FR": 'Time feature',
    "Lag_J1_USD_EUR_SPOT": 'Change',
    "Lag_J1_GBP_EUR_SPOT": 'Change',
    "Lag_J2_Exchange_FR_DE": "Exchange",
    "Lag_J2_Exchange_FR_TOT": "Exchange",

}

PALETTE_DIC = {'Change' : '#fbafe4',
 'Commodity price' : '#949494',
 'Exchange' : '#ece133',
 'Nuclear_availability' : "#029e73",
 'Prev_Residual_Load' : '#d55e00',
 'Production J-2' : "#de8f05",
 'Production J-7' : "#EEC782",
 'SpotPrice J-1' :'#0173b2',
 'SpotPrice J-7' : '#80B9D8',
 'Time feature' : "#cc78bc"}


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

#### models.evaluate ####


def plot_time_series(df, model_name, start_date, end_date, quantile=True, list_quantiles=None):
    list_quantiles = sorted(list(df['quantile'].unique())) if (quantile and list_quantiles is None) else list_quantiles
    data = df[(df.Date >= start_date) & (df.Date <= end_date)].reset_index(drop=True)
    data.sort_values(by='Time', inplace=True)
    plt.figure(figsize=(20, 10))

    if quantile:
        pal = sns.color_palette('vlag', n_colors=len(list_quantiles))
        if isinstance(model_name, list):
            styles = ['solid', 'dotted', 'dashed', 'dashdot']
            for j, model in enumerate(model_name):
                for i, q in enumerate(sorted(list_quantiles)):
                    data_to_plot = data[(data['quantile'] == q) & (data.model_name == model)]
                    plt.plot(data_to_plot.Time,
                             data_to_plot.SpotPrice_pred,
                             label=f'{model}_{q}',
                             color=pal[i],
                             linewidth=1,
                             alpha=0.7,
                             linestyle=styles[j]
                             )

        else:
            for i, q in enumerate(sorted(list_quantiles)):
                data_to_plot = data[(data['quantile'] == q) & (df.model_name == model_name)]
                plt.plot(data_to_plot.Time,
                         data_to_plot.SpotPrice_pred,
                         label=q,
                         color=pal[i],
                         linewidth=1,
                         alpha=0.7
                         )

    else:
        data_to_plot = data.copy()
        plt.plot(data.Time,
                 data.SpotPrice_pred,
                 label='Price Prediction',
                 color='r',
                 linewidth=1.5,
                 alpha=0.7,
                 linestyle='--'
                 )

    plt.scatter(data_to_plot.Time,
                data_to_plot.SpotPrice,
                label='True Price',
                color='k',
                s=5
                )
    plt.plot(data_to_plot.Time,
             data_to_plot.SpotPrice,
             label='True Price',
             color='k',
             linewidth=0.4
             )
    plt.legend()
    plt.title(model_name)
    plt.show()


def reorder_quantiles(df):
    df = df.sort_values(by=['Date', 'Hour', 'model', 'quantile'])
    dfp = pd.pivot_table(df, values='SpotPrice_pred', index=['model', 'Date', 'Hour', 'SpotPrice'],
                         columns='quantile')
    dfp.columns = np.sort(dfp.columns)
    dfp.iloc[:, :] = np.sort(dfp, axis=1)
    df = pd.melt(dfp.reset_index(), id_vars=['model', 'Date', 'Hour', 'SpotPrice'], var_name='quantile',
                 value_name='SpotPrice_pred')
    df = df.sort_values(by=['Date', 'Hour', 'model', 'quantile'])
    return df


# TODO : Optimize speed
def evaluate_quantile_pred(df_pred, alpha, start=None, end=None, plot=False):
    start = df_pred.Date.min() if start is None else start
    end = df_pred.Date.max() if end is None else end
    df_quantile_scores = pd.DataFrame()

    for a in tqdm(alpha):
        assert a != 0, "Cannot compute quantile score for median prediction, Please use RMSE"
        q_up, q_down = round(100 * (a + 1) / 2, 2), round(100 * (1 - a) / 2, 2)
        for model_name in df_pred['model_name'].unique():
            df_to_score = df_pred[(df_pred.model_name == model_name) &
                                  (df_pred['quantile'].isin([q_up, q_down])) &
                                  (df_pred.Date >= start) &
                                  (df_pred.Date <= end)
                                  ]
            assert len(df_to_score) > 0, "The dataframe to score has length zero, check the quantile values"

            y_true = df_to_score[df_to_score['quantile'] == q_up].SpotPrice.reset_index(drop=True)
            y_high = df_to_score[df_to_score['quantile'] == q_up].SpotPrice_pred.reset_index(drop=True)
            y_low = df_to_score[df_to_score['quantile'] == q_down].SpotPrice_pred.reset_index(drop=True)

            emp_cov = empirical_coverage(y_true, y_low, y_high)
            wr_score = wrinkler_score(y_true, y_low, y_high, alpha=a)
            # unc_cov = unconditional_coverage(y_true, y_low, y_high, alpha=a)
            int_width = interval_width(y_low, y_high)

            df_quantile_scores = df_quantile_scores.append(
                pd.DataFrame({
                    'model_name': [model_name],
                    'alpha': [a],
                    'empirical_coverage': [emp_cov],
                    'wrinkler_score': [wr_score],
                    # 'unconditional_coverage': [unc_cov], # TODO : Fix this score
                    'interval_width': [int_width]
                })
            )

    if plot:
        fig, axes = plt.subplots(1, 3)
        for i, score in enumerate(['empirical_coverage', 'wrinkler_score', 'interval_width']):

            sns.lineplot(
                data=df_quantile_scores.rename(columns={'alpha': 'target_coverage'}).reset_index(),
                x='target_coverage',
                y=score,
                hue='model_name',
                style='model_name',
                markers=True,
                ax=axes[i])

            if i < 2:
                axes[i].legend([])
            else:
                axes[i].legend(loc='upper right', bbox_to_anchor=(1.5, 1))

            if score == 'empirical_coverage':
                axes[i].plot([0, 1], [0, 1], c='k')

        fig.set_size_inches((30, 10))

    return df_quantile_scores


def talagrand_plot(df_pred, quantiles, start=None, end=None, plot=False):
    start = df_pred.Date.min() if start is None else start
    end = df_pred.Date.max() if end is None else end

    dfte = df_pred[(df_pred.Date >= start) & ((df_pred.Date < end))]
    dfte = dfte.groupby(['Date', 'Hour', 'model_name', 'SpotPrice', 'quantile']).SpotPrice_pred.mean().reset_index()
    dfp = pd.pivot(dfte, index=['Date', 'Hour', 'model_name', 'SpotPrice'], columns=['quantile'],
                   values=['SpotPrice_pred']).reset_index()
    level_two = dfp.columns.get_level_values(1)
    dfp.columns = dfp.columns.get_level_values(0)
    dfp.columns = [f'{l0}_{l1}' if l1 != '' else l0 for l0, l1 in zip(dfp.columns, list(level_two.astype(str)))]
    dfp[f'interval'] = np.nan
    for i, quantile in enumerate(quantiles[:-1]):
        if i == 0:
            dfp.loc[(dfp.SpotPrice >= dfp[f'SpotPrice_pred_{quantile}']) & (dfp.SpotPrice < dfp[
                f'SpotPrice_pred_{quantile + 5}']), 'interval'] = f'0{quantile}-{quantile + 5}'
        else:
            dfp.loc[(dfp.SpotPrice >= dfp[f'SpotPrice_pred_{quantile}']) & (dfp.SpotPrice < dfp[
                f'SpotPrice_pred_{quantile + 5}']), 'interval'] = f'{quantile}-{quantile + 5}'
        dfp.loc[(dfp.SpotPrice < dfp[f'SpotPrice_pred_{quantile}']) & (
                dfp.SpotPrice >= dfp[f'SpotPrice_pred_{quantile + 5}']), 'interval'] = f'overlap'

    dfp.loc[(dfp.SpotPrice >= dfp[f'SpotPrice_pred_95']), 'interval'] = f'95-100'
    dfp.loc[(dfp.SpotPrice < dfp[f'SpotPrice_pred_5']), 'interval'] = f'0-5'

    dfg = pd.DataFrame(dfp.groupby(['model_name']).interval.value_counts())
    dfg.columns = ['count']
    dfg.reset_index(inplace=True)

    dfg['Number of true price in predicted interval'] = dfg['count'] / (dfg['count'].sum()) * len(
        df_pred.model_name.unique())

    if plot:
        sns.barplot(
            data=dfg.sort_values(by=['interval', 'model_name']),
            x='interval',
            y='Number of true price in predicted interval',
            hue='model_name'
        )

        plt.xticks(rotation=60)
        plt.title('Talagrand plot')
        plt.legend(bbox_to_anchor=(1.7, 1), loc='upper right')
        plt.show()

    return dfg


def evaluate_pinball(df):
    diff = df.SpotPrice - df.SpotPrice_pred
    sign = (diff >= 0).astype(int)
    df['pinball_loss'] = (df['quantile'] / 100) * sign * diff - (1 - (df['quantile'] / 100)) * (1 - sign) * diff
    df['pinball_loss'] = df['pinball_loss'].astype(float)

    return df


def pinball_plot(df, start=None, end=None):
    start = df.Date.min() if start is None else start
    end = df.Date.max() if end is None else end

    df_pred = df[(df.Date >= start) & (df.Date <= end)]
    dfg = df_pred.groupby(['model_name', 'quantile']).pinball_loss.mean().reset_index()
    dfg = dfg.sort_values(by='model_name')
    plt.figure(figsize=(15, 7))
    sns.barplot(data=dfg, x='quantile', y='pinball_loss', hue='model_name')

    return dfg


def special_split(df, target_sep, list_sep):
    if not target_sep == 'model_type':
        col = df['model'].str.split(target_sep).str[1].fillna('')
    else:
        col = df['model']

    for sep in list_sep:
        if target_sep == sep:
            pass
        else:
            col = col.str.split(sep).str[0]

    return col


def serial_special_split(df, list_sep):
    df['model_type'] = special_split(df, 'model_type', list_sep)
    for target_sep in list_sep:
        df[target_sep] = special_split(df, target_sep, list_sep)
    return df


def format_results(df, quantile_prediction=True, list_sep=None):
    if quantile_prediction:
        df = reorder_quantiles(df)

    list_sep = ['PER', 'CAL', 'GAMMA', 'TYPE'] if (list_sep is None) else list_sep

    df['model_type'] = special_split(df, 'model_type', list_sep)
    for target_sep in list_sep:
        df[target_sep] = special_split(df, target_sep, list_sep)
    df['model_name'] = df['model'].copy()

    df['Time'] = pd.to_datetime(df.Date + "-" + df.Hour.astype(str), format="%Y-%m-%d-%H")

    return df


############################ Mean regression #####################################"


def plot_time_series_mreg(df, model_name, start_date, end_date, quantile=True, list_quantiles=None):
    list_quantiles = sorted(list(df['quantile'].unique())) if (quantile and list_quantiles is None) else list_quantiles
    data = df[(df.Date >= start_date) & (df.Date <= end_date)].reset_index(drop=True)
    data.sort_values(by='Time', inplace=True)
    plt.figure(figsize=(20, 10))

    data_to_plot = data[data.model_name == model_name].copy()
    plt.plot(data_to_plot.Time,
             data_to_plot.SpotPrice_pred,
             label='Price Prediction',
             color='r',
             linewidth=1.5,
             alpha=0.7,
             linestyle='--'
             )

    plt.scatter(data_to_plot.Time,
                data_to_plot.SpotPrice,
                label='True Price',
                color='k',
                s=5
                )
    plt.plot(data_to_plot.Time,
             data_to_plot.SpotPrice,
             label='True Price',
             color='k',
             linewidth=0.4
             )
    plt.legend()
    plt.title(model_name)
    plt.show()


def evaluate_mean_prediction(df, start=None, end=None, year=True):
    # Compute random error :
    df_ = df[['Date', 'Hour', 'SpotPrice']].drop_duplicates()
    df_['lagged_SpotPrice'] = df_['SpotPrice'].shift(24 * 7)
    df_ = df_[-df_.lagged_SpotPrice.isna()]
    naive_error = np.mean(np.abs(df_['lagged_SpotPrice'] - df_['SpotPrice']))

    start = df.Date.min() if start is None else start
    end = df.Date.max() if end is None else end
    dfl = df.loc[(df.Date >= start) & (df.Date <= end)]
    dfl['year'] = pd.to_datetime(df.Date).dt.year
    dfl['mse'] = np.square(dfl.SpotPrice_pred - dfl.SpotPrice)
    dfl['mae'] = np.abs(dfl.SpotPrice_pred - dfl.SpotPrice)
    dfl['rmae'] = np.abs(dfl.SpotPrice_pred - dfl.SpotPrice) / naive_error
    if year:
        dfg = dfl.groupby(['model_type', 'PER','year'])['mse', 'mae', 'rmae', 'time'].mean()
    else:
        dfg = dfl.groupby(['model_type', 'PER'])['mse', 'mae', 'rmae', 'time'].mean()
    
    dfg['rmse'] = np.sqrt(dfg['mse'])
    dfg = dfg.reset_index()
    dfg['PER'] = pd.Categorical(dfg['PER'], categories=['90', '180', '270', '360', '720', '1080', 'None'], ordered=True)
    if year:
        dfg.sort_values(['model_type', 'PER', 'year'], inplace=True)
        dfg = dfg.set_index(['model_type', 'PER', 'year'])
    else:
        dfg.sort_values(['model_type', 'PER'], inplace=True)
        dfg = dfg.set_index(['model_type', 'PER'])

    return dfg


#### models.metrics ####


def pinball_loss(y_true, y_pred, alpha=0.5):
    diff = y_true - y_pred
    sign = (diff >= 0).astype(diff.dtype)
    loss = alpha * sign * diff - (1 - alpha) * (1 - sign) * diff
    
    return loss


def mean_pinball_loss(y_true, y_pred, alpha=0.5):
    loss = pinball_loss(y_true, y_pred, alpha=alpha)
    avg_loss = average(loss)
    
    return avg_loss


def unconditional_coverage(y_true, y_low, y_high, alpha=0.1):
    emp_cov = empirical_coverage(y_true, y_low, y_high, mean_=False)
    n1 = np.sum(emp_cov)
    n0 = len(emp_cov) - n1
    pi = n1 / (n1 + n0)
    c = (1-alpha)

    if n1 < 0.001:
        unconditional_cov = 0
    else:
        unconditional_cov = -2 * (n0 * log(1-c) + n1 * log(c) - n0 * log(1-pi) - n1 * log(pi))
    
    return unconditional_cov


def kupiec_test(y_true, y_low, y_high, alpha=0.1, conf_level=0.95):

    N = len(y_true)
    
    uc = unconditional_coverage(y_true, y_low, y_high, alpha)
    kupiec_test = chi2.cdf(uc, 1)
    
    if kupiec_test < conf_level:
        result = 1 # Fail to reject H0
    else:
        result = 0 # Reject H0

    return kupiec_test, result



def DM(y_true, quantile_forecast_1, quantile_forecast_2, tab_quantile):
    """
    Diebold-Mariano test on CRPS

    Parameters :
    ------------

    y_true : numpy array, either (N_obs) or (N_hours, N_obs)
    quantile_forecast_1 : numpy array, either (N_quantiles, N_obs) or (N_hours, N_quantiles, N_obs)
    """
    if y_true.ndim > 1:
        # Check dimensions :
        if y_true.shape[1] != quantile_forecast_1.shape[2] or \
         quantile_forecast_2.shape[2] != quantile_forecast_1.shape[2] or \
         y_true.shape[0] != quantile_forecast_1.shape[0] or \
         y_true.shape[0] != quantile_forecast_2.shape[0]:
            raise ValueError('The three time series must have the same shape')

        d = np.zeros(y_true.shape)
        # We have all hours
        for h in range(y_true.shape[0]):
            errors_pred_1 = crps(y_true[h], quantile_forecast_1[h], tab_quantile)
            errors_pred_2 = crps(y_true[h], quantile_forecast_2[h], tab_quantile)

            d[h] = np.abs(errors_pred_1) - np.abs(errors_pred_2)
        
            # Computing the loss differential size
        
        delta = np.sum(d, axis=0)

    else:
        # Only 1 hour

        # Checking that all time series have the same shape
        if y_true.shape[0] != quantile_forecast_1.shape[1] or quantile_forecast_2.shape[1] != quantile_forecast_1.shape[1]:
            raise ValueError('The three time series must have the same shape')

        # Computing the errors of each forecast
        errors_pred_1 = crps(y_true, quantile_forecast_1, tab_quantile)
        errors_pred_2 = crps(y_true, quantile_forecast_2, tab_quantile)

        # We choose L1 norm
        delta = np.abs(errors_pred_1) - np.abs(errors_pred_2)
        
    # Computing the loss differential size
    N = delta.shape[0]

    # Computing the test statistic
    mean_d = np.mean(delta, axis=0)
    var_d = np.var(delta, ddof=0, axis=0)
    DM_stat = mean_d / np.sqrt((1 / N) * var_d)

    p_value = 1 - norm.cdf(DM_stat)

    return p_value


def empirical_coverage(y_true, y_low, y_high, mean_=True):
    coverage = (y_true <= y_high).astype(int) * (y_true >= y_low).astype(int)
    if mean_:
        coverage = mean(coverage)
    return coverage


def wrinkler_score(y_true, y_low, y_high, alpha=0.1, mean_=True):
    '''
    Computes Wrinkler score
    :param mean_:
    :param y_true:
    :param y_low:
    :param y_high:
    :param alpha:
    :return: array - Wrinkler score
    '''
    wrinkler = zeros(y_true.shape[0])
    delta = y_high - y_low
    for i, (p, u, d) in enumerate(zip(y_true, y_low, y_high)):
        if d <= p <= u:
            wrinkler[i] = delta[i]
        elif p < d:
            wrinkler[i] = delta[i] + 2/alpha * (d - p)
        else:
            wrinkler[i] = delta[i] + 2/alpha * (p - u)
    if mean_:
        wrinkler = mean(wrinkler)
    return wrinkler


def interval_width(y_low, y_high, mean_=True):
    width = y_high - y_low
    if mean_:
        width = mean(width)
    return width

def crps(y_true, quantile_forecast, tab_quantile):
    '''
    CRPS function - Computes the CRPS score thanks to the reformulation as an integral on the pinball loss.
    Uses Riemann integration approximation to computes the CRPS.
    
    ** This function computes CRPS term by term **
    '''
    N = len(y_true)
    riemann_sum = np.zeros(N)
    for i in range(1, len(tab_quantile)):
        step_size = (tab_quantile[i] - tab_quantile[i-1]) 
        losses = pinball_loss(y_true, quantile_forecast[i], alpha=tab_quantile[i])
        riemann_sum += step_size * losses
    
    return riemann_sum


def mean_crps(y_true, quantile_forecast, tab_quantile):
    '''
    CRPS function - Computes the CRPS score thanks to the reformulation as an integral on the pinball loss.
    Uses Riemann integration approximation to computes the CRPS
    '''
    return average(crps(y_true, quantile_forecast, tab_quantile))
