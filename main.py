import pandas as pd
from preprocessing.variable_selection import recursive_feature_elimination
from preprocessing.preprocessing import date_split
from catboost import CatBoostRegressor


def load_and_process_data():
    df = pd.read_csv('data/data_processed.csv')
    df = df.drop(columns=['Unnamed: 0'])
    df['Time'] = pd.to_datetime(df.Date + "-" + df.Hour.astype(str), format="%Y-%m-%d-%H")
    df = df[df.Date > "2018-01-07"]

    # for i, feat in enumerate(df.columns):
    #    print(i, ' : ', feat)

    features = ["Prev_J2_Load", "Prev_J1_Solar", "Prev_J1_WindOnshore", "M1_Coal", "M1_Oil"]
    features += list(df.columns[50:70])  # ignore lag J2 & J7 Solar
    features += list(df.columns[72:88])
    features += ["PublicHoliday", "SchoolHoliday_FR", "weekday", "toy_cos", "Ponts_FR", 'toy_sin', 'hour_cos',
                 'hour_sin']

    df_train, df_test = date_split(df, date="2020-12-31")
    df_train_sep, df_val = date_split(df_train, date="2019-12-31")

    return df_train, df_test, df_train_sep, df_val, features


def split(df_train, df_test, df_train_sep, df_val, features):
    X_train, y_train = df_train[features], df_train["SpotPrice"]
    X_train_sep, y_train_sep = df_train_sep[features], df_train_sep["SpotPrice"]
    X_val, y_val = df_val[features], df_val["SpotPrice"]
    X_test, y_test = df_test[features], df_train["SpotPrice"]

    return X_train, y_train, X_train_sep, y_train_sep, X_val, y_val, X_test, y_test


def main():

    df_train, df_test, df_train_sep, df_val, features = load_and_process_data()
    X_train, y_train, X_train_sep, y_train_sep, X_val, y_val, X_test, y_test = split(df_train, df_test, df_train_sep,
                                                                                     df_val, features)
    model = CatBoostRegressor(verbose=0)
    models, results = recursive_feature_elimination(X_train, y_train, df_train.Date, model, n_features=48, n_features_min=46)


if __name__ == '__main__':
    main()
