from pandas import to_datetime, read_csv, merge, concat, pivot
from datetime import timedelta
from vars_preprocessing import DIC_LAGS, PIB_PERC, PERC_POP_ZONE, LIST_FOREIGN_COUNTRIES, DIC_COL_INDISPO
from numpy import nan, sin, cos, pi


def load_process_enrich_data():
    df = read_csv('data-raw/Data_2018_2021.csv')
    df.Date = to_datetime(df.Date)
    df.sort_values(by=['Date', 'Hour'], inplace=True)

    # Nans
    df = complete_nans(df)

    # Export & Import
    for nat in LIST_FOREIGN_COUNTRIES:
        df[f'Exchange_FR_{nat}'] = df[f'Import_FR_{nat}'] + df[f'Export_FR_{nat}']
        df = df.drop(columns=[f'Import_FR_{nat}', f'Export_FR_{nat}'])

    # Lags
    df = apply_lags(df)

    # School & public holidays :
    df = add_public_holidays(df)

    # Indisponibilités
    df = add_indisponibilites(df)

    # Week days & toy encoding & Hours encoding
    df['weekday'] = df.Date.dt.dayofweek

    df['toy_sin'] = sin(2 * pi * (df.Date.dt.dayofyear / 365.))
    df['toy_cos'] = cos(2 * pi * (df.Date.dt.dayofyear / 365.))
    df['toy'] = df.Date.dt.dayofyear / 365.

    df['hour_sin'] = sin(2 * pi * (df.Hour / 24))
    df['hour_cos'] = cos(2 * pi * (df.Hour / 24))

    return df


def complete_nans(data):
    """
    Completes Nans value in Data according to description
    """

    df = data.copy()

    cols_interp = ["Load", "Biomass", 'Fossil_Gas', 'Fossil_Hard_Coal', 'Fossil_Oil', 'Hydro_Pumped_Storage',
                   'Hydro_Run-of-river_and_poundage', 'Hydro_Water_Reservoir', 'Nuclear',
                   'Solar', 'Waste', 'Wind_Onshore', "IT_SpotPrice", "Import_FR_BE"]
    cols_ffill = ["M1_Oil", "M1_Coal", "USD_EUR_SPOT", "GBP_EUR_SPOT"]
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
    df.loc[df.Date == to_datetime('2020-03-03'), 'Prev_J2_Load'] = 0.5 * (
            df.loc[df.Date == to_datetime('2020-03-03') - timedelta(7), 'Prev_J2_Load'].to_numpy() +
            df.loc[df.Date == to_datetime('2020-03-03') + timedelta(7), 'Prev_J2_Load'].to_numpy())

    return df


def apply_lags(df, dic_lags=DIC_LAGS):
    """
    Apply lags on the different variables
    """
    df = df.sort_values(by=['Date', 'Hour'])
    for col in dic_lags.keys():
        list_lags = dic_lags[col]
        for lag in list_lags:
            df[f"Lag_J{lag}_{col}"] = df[col].shift(24 * lag)

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

    # Jour fériés étrangers :
    df_jf = read_csv('data-raw/jours_feries.csv')
    df_jf.rename(columns={"countryOrRegion": "country", "countryRegionCode": "country_code", 'date': "Date"},
                 inplace=True)
    df_jf = df_jf[~(df_jf.country == 'France')]
    # Beaucoup trop de jours pour UK : on ne prend que ceux où il y a de l'arrêt
    df_jf = df_jf[df_jf.isPaidTimeOff.isin([nan, True])]
    df_jf['Date'] = to_datetime(df_jf.Date)

    dfm = merge(df[["Date", "Hour"]], df_jf[['country_code', 'Date']], on='Date', how='left')
    dfm['has_hd'] = (~dfm.country_code.isna()).astype(int)

    dfm = dfm.pivot(index=['Date', 'Hour'], columns='country_code', values='has_hd')
    dfm.reset_index(inplace=True)
    dfm = dfm.fillna(0)
    dfm = dfm[['Date', 'Hour', 'BE', 'DE', 'GB', 'IT']]

    dfm = dfm.rename(columns={
        col: f"PublicHoliday_{col}" for col in ['BE', 'DE', 'GB', 'IT']
    })
    df = merge(df, dfm, on=['Date', 'Hour'], how='left')

    # Fusion des variables de vacances\jour fériés
    somme = 0
    for col in ['BE', 'DE', 'FR', 'GB', 'IT']:
        somme += df[f'PublicHoliday_{col}'] * PIB_PERC[col]
    df['PublicHoliday'] = somme

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
    df = df[df.Time.dt.year.isin([y for y in range(2018, 2022)])]
    df.rename(columns=DIC_COL_INDISPO, inplace=True)
    df['Unknown_availability'] = df['all'] - df['sum']
    df = concat([data, df.drop(columns=['all', 'sum', 'Time', 'Hour', 'Date']).reset_index(drop=True)], axis=1)

    # Variables constantes presque partout
    df.drop(columns=["Marine_availability", "Unknown_availability", "Other_availability"], inplace=True)

    # Variables négligeables
    df.drop(columns=['Biomass_availability'], inplace=True)
    return df


def pivot_hour(data, features):
    dfh = pivot(
        data=data,
        index=['Date'],
        values=features,
        columns=['Hour']
    ).reset_index()

    level_two = dfh.columns.get_level_values(1)
    dfh.columns = dfh.columns.get_level_values(0)
    dfh.columns = [f'{col}_H{hour}' for col, hour in zip(dfh.columns, list(level_two.astype(str)))]

    return dfh


def main():
    df = load_process_enrich_data()
    df.to_csv('data/data_processed.csv', index=False)


if __name__ == "__main__":
    main()
    print("Data successfully treated and saved !")
