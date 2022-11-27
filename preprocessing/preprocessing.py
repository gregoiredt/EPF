from pandas import to_datetime, read_csv, merge, concat, pivot
from datetime import timedelta
from .vars_preprocessing import DIC_LAGS, PIB_PERC, PERC_POP_ZONE, LIST_FOREIGN_COUNTRIES, DIC_COL_INDISPO, \
    DIC_VAR_CLASSIF
from numpy import nan, sin, cos, pi
import pickle
from loguru import logger


def load_process_enrich_data():
    df = read_csv('data-raw/Data_2016_2021.csv')
    df.Date = to_datetime(df.Date)
    df.sort_values(by=['Date', 'Hour'], inplace=True)
    # Nans
    df = complete_nans(df)

    # Export & Import
    df[f'Exchange_FR_TOT'] = 0
    for nat in LIST_FOREIGN_COUNTRIES:
        df[f'Exchange_FR_{nat}'] = df[f'Import_FR_{nat}'] + df[f'Export_FR_{nat}']
        df[f'Exchange_FR_TOT'] += df[f'Exchange_FR_{nat}']
        df = df.drop(columns=[f'Import_FR_{nat}', f'Export_FR_{nat}'])

    # School & public holidays :
    df = add_public_holidays(df)

    # Indisponibilités
    df = add_indisponibilites(df)

    # Week days & toy encoding & Hours encoding
    df['weekday'] = df.Date.dt.dayofweek

    df['toy_sin'] = sin(2 * pi * (df.Date.dt.dayofyear / 365.))
    df['toy_cos'] = cos(2 * pi * (df.Date.dt.dayofyear / 365.))
    df['toy'] = df.Date.dt.dayofyear / 365.

    # df['hour_sin'] = sin(2 * pi * (df.Hour / 24))
    # df['hour_cos'] = cos(2 * pi * (df.Hour / 24))

    dic_var = DIC_VAR_CLASSIF.copy()

    # Lags
    df = apply_lags(df, dic_var)

    # Demande residuelle :
    df = _compute_demande_residuelle(df, dic_var)

    # Hourly separation
    df = pivot_hour(df, dic_var, only_central=True, only_exo=False)
    df = normalize_prod(df)

    # Add clock
    df['clock'] = (to_datetime(df['Date']) - to_datetime(df['Date'].min())).dt.days

    # select variables

    all_features = ["Date", 'Hour', 'SpotPrice']
    for key in dic_var.keys():
        all_features += dic_var[key]
    df = df[all_features]

    return df, dic_var


def _compute_demande_residuelle(data, dic_var):
    assert "Prev_J2_Load" in dic_var['exogenous_central_hourly_valid'], "Prev_J2_Load not present"
    assert "Prev_J1_Solar" in dic_var['exogenous_central_hourly_valid'], "Prev_J1_Solar not present"
    assert "Prev_J1_WindOnshore" in dic_var['exogenous_central_hourly_valid'], "Prev_J1_WindOnshore not present"
    assert "Lag_J2_Hydro_Run-of-river_and_poundage" in dic_var['exogenous_additional_hourly_J2'], "Lag_J2_Hydro_Run-of-river_and_poundage not present"

    name_var = 'Prev_Residual_Load'
    data[name_var] = data["Prev_J2_Load"] - data["Prev_J1_Solar"] - data["Prev_J1_WindOnshore"] - data[
        "Lag_J2_Hydro_Run-of-river_and_poundage"]

    dic_var["exogenous_central_hourly_valid"] = [var for var in dic_var["exogenous_central_hourly_valid"]
                                                 if not(var in [ "Prev_J2_Load", "Prev_J1_Solar", "Prev_J1_WindOnshore"])]
    dic_var["exogenous_central_hourly_valid"] += [name_var]
    dic_var["exogenous_additional_hourly_J2"] = [col_name for col_name in dic_var["exogenous_additional_hourly_J2"] if \
                                                 not (col_name == "Lag_J2_Hydro_Run-of-river_and_poundage")]

    return data


def complete_nans(data):
    """
    Completes Nans value in Data according to description
    """

    df = data.copy()

    cols_interp = ["Load", "Biomass", 'Fossil_Gas', 'Fossil_Hard_Coal', 'Fossil_Oil', 'Hydro_Pumped_Storage',
                   'Hydro_Run-of-river_and_poundage', 'Hydro_Water_Reservoir', 'Nuclear',
                   'Solar', 'Waste', 'Wind_Onshore', "IT_SpotPrice", "Import_FR_BE", 'Import_FR_IT_North']
    cols_ffill = ["M1_Oil", "M1_Coal", "USD_EUR_SPOT", "GBP_EUR_SPOT", 'Export_FR_IT_North']
    cols_interp_per_hour = ['Prev_J1_Solar', 'Prev_J1_WindOnshore']
    cols_gb = ["Export_FR_GB", "Import_FR_GB"]

    # Linear interpolation on TS
    df[cols_interp] = df[cols_interp].interpolate(method='linear')
    df[cols_gb] = df[cols_gb].interpolate(method='linear', limit=2)

    # Filling with last observed value
    df[cols_ffill] = df[cols_ffill].ffill()
    df[cols_ffill] = df[cols_ffill].bfill()

    # Linear interpolation on the daily level
    df = df.sort_values(by=['Hour', 'Date'])
    df[cols_interp_per_hour] = df[cols_interp_per_hour].interpolate(method='linear')
    df[cols_gb] = df[cols_gb].interpolate(method='linear')
    df = df.sort_values(by=['Date', 'Hour'])

    # Moving average
    for date in ['2017-06-06', '2017-06-07', '2017-06-08', '2020-03-03']:
        df.loc[df.Date == to_datetime(date), 'Prev_J2_Load'] = 0.5 * (
                df.loc[df.Date == to_datetime(date) - timedelta(7), 'Prev_J2_Load'].to_numpy() +
                df.loc[df.Date == to_datetime(date) + timedelta(7), 'Prev_J2_Load'].to_numpy())

    return df


def apply_lags(df, dic_var, dic_lags=DIC_LAGS):
    """
    Apply lags on the different  and updates dictionary of variable
    """
    df = df.sort_values(by=['Date', 'Hour'])
    categories = list(dic_var.keys()).copy()
    for cat in categories:
        cols = dic_var[cat]
        # We consider that applying lags makes the feature valid
        cat_name = cat if not ('valid' in cat) else ''.join([w + '_' if i < (len(cat.split('_')[:-1])-1) else w
                                                             for i, w in enumerate(cat.split('_')[:-1]) ])
        for col in cols:
            if col in dic_lags.keys():
                list_lags = dic_lags[col]
                for lag in list_lags:
                    df[f"Lag_J{lag}_{col}"] = df[col].shift(24 * lag)

                    lag_name = 2 if lag == 1 else lag  # Un lag de J-1 est considéré comme obtenu à J-2

                    if f'{cat_name}_J{lag_name}' not in dic_var.keys():
                        dic_var[f'{cat_name}_J{lag_name}'] = []
                    dic_var[f'{cat_name}_J{lag_name}'].append(f"Lag_J{lag}_{col}")
            else:
                pass

    return df


def add_public_holidays(df):
    # Jour fériés français
    df_jf_fr = read_csv('data-raw/fr_jour_feries.csv')
    df_jf_fr['Date'] = to_datetime(df_jf_fr.date)
    df_jf_fr.drop(columns=['Annee', 'JourSemaine', 'date'], inplace=True)
    df = merge(df, df_jf_fr, on='Date', how='left')
    df.rename(columns={
        "JourFerie": "PublicHoliday_FR",
        "Ponts": "Ponts_FR",
        "holiday_A": "SchoolHoliday_FR_A",
        "holiday_B": "SchoolHoliday_FR_B",
        "holiday_C": "SchoolHoliday_FR_C",
    }, inplace=True)

    # # Jour fériés étrangers :
    # df_jf = read_csv('data-raw/jours_feries.csv')
    # df_jf.rename(columns={"countryOrRegion": "country", "countryRegionCode": "country_code", 'date': "Date"},
    #              inplace=True)
    # df_jf = df_jf[~(df_jf.country == 'France')]
    # # Beaucoup trop de jours pour UK : on ne prend que ceux où il y a de l'arrêt
    # df_jf = df_jf[df_jf.isPaidTimeOff.isin([nan, True])]
    # df_jf['Date'] = to_datetime(df_jf.Date)
    #
    # dfm = merge(df[["Date", "Hour"]], df_jf[['country_code', 'Date']], on='Date', how='left')
    # dfm['has_hd'] = (~dfm.country_code.isna()).astype(int)
    #
    # dfm = dfm.pivot(index=['Date', 'Hour'], columns='country_code', values='has_hd')
    # dfm.reset_index(inplace=True)
    # dfm = dfm.fillna(0)
    # dfm = dfm[['Date', 'Hour', 'BE', 'DE', 'GB', 'IT']]
    #
    # dfm = dfm.rename(columns={
    #     col: f"PublicHoliday_{col}" for col in ['BE', 'DE', 'GB', 'IT']
    # })
    # df = merge(df, dfm, on=['Date', 'Hour'], how='left')
    #
    # # Fusion des variables de vacances\jour fériés
    # somme = 0
    # for col in ['BE', 'DE', 'FR', 'GB', 'IT']:
    #     somme += df[f'PublicHoliday_{col}'] * PIB_PERC[col]
    # df['PublicHoliday'] = somme

    somme = 0
    for zone in ['A', 'B', 'C']:
        somme += df[f'SchoolHoliday_FR_{zone}'] * PERC_POP_ZONE[zone]
    df['SchoolHoliday_FR'] = somme

    return df


def date_split(data, date):
    assert (date <= data.Date.max()) and (date >= data.Date.min()), "The date is not valid for the given data"
    data_train, data_test = data[data.Date <= date], data[data.Date > date]
    return data_train, data_test


def add_indisponibilites(data):
    df = read_csv('data-raw/Indispos_2016_2021.csv', index_col=0)
    df['Time'] = to_datetime(df.index.to_series().astype(str).apply(lambda x: x.split('+')[0]))
    df['Date'] = df.Time.dt.date
    df['Hour'] = df.Time.dt.hour
    df.reset_index(drop=True, inplace=True)
    df = df[df.Time.dt.year.isin(data.Date.dt.year.unique())]
    df.rename(columns=DIC_COL_INDISPO, inplace=True)
    df['Unknown_availability'] = df['all'] - df['sum']
    df = concat([data, df.drop(columns=['all', 'sum', 'Time', 'Hour', 'Date']).reset_index(drop=True)], axis=1)

    # Variables constantes presque partout
    df.drop(columns=["Marine_availability", "Unknown_availability", "Other_availability"], inplace=True)

    # Variables négligeables
    df.drop(columns=['Biomass_availability'], inplace=True)

    return df


def normalize_prod(data):
    prod_features = [
        # "Fossil_Gas",
        # "Fossil_Hard_Coal",
        # "Fossil_Oil",
        # "Hydro_Pumped_Storage",
        # "Hydro_Run-of-river_and_poundage",
        # "Hydro_Water_Reservoir",
        "Nuclear"
    ]

    for feature in prod_features:
        lags = DIC_LAGS[feature]
        for lag in lags:
            data[f'Lag_J{lag}_{feature}'] = data[f'Lag_J{lag}_{feature}'] / data[f'{feature}_availability']

    return data


def pivot_hour(data, dic_var, only_central=True, only_exo=True):
    """
    Places the hour in the distribution
    :param only_exo: bool - Retrieve only exogenous variables
    :param only_central: bool - Retrieve only central variables
    :param data: DataFrame - original data
    :param dic_var: Dic - dictionnary of variable names
    :return: DataFrame - dataframe with new columns
    """
    keys = dic_var.keys()
    if only_exo:
        keys = [k for k in keys if "exogenous" in k]
    if only_central:
        keys = [k for k in keys if "central" in k]
    keys = [k for k in keys if "hourly" in k]
    for key in keys:
        features = dic_var[key]
        dfh = pivot(
            data=data,
            index=['Date'],
            values=features,
            columns=['Hour']
        ).reset_index()

        level_two = dfh.columns.get_level_values(1)
        dfh.columns = dfh.columns.get_level_values(0)
        dfh.columns = [f'{col}_H{hour}' for col, hour in zip(dfh.columns, list(level_two.astype(str)))]
        dic_var[key] = [f'{col}' for col, hour in zip(dfh.columns, list(level_two.astype(str))) if "Date" not in col]
        dfh.rename(columns={"Date_H": "Date"}, inplace=True)

        if "SpotPrice" in features:
            # On veut garder la target
            features = list(set(features) - {"SpotPrice"})
        data = merge(data.drop(columns=features), dfh, on=['Date'])

    return data


def process_and_save_data(filename_extension):
    df, dic_var = load_process_enrich_data()

    a_file = open(f"data/dic_var_{filename_extension}.pkl", "wb")
    pickle.dump(dic_var, a_file)
    a_file.close()

    df.to_csv(f'data/data_cleaned_{filename_extension}.csv', index=False)
    logger.info('Successfully saved data and variables !')

