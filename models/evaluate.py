from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from .metrics import empirical_coverage, wrinkler_score, interval_width


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
