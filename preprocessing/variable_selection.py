import pandas as pd
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from numpy import mean, std


import matplotlib.pyplot as plt


def feature_importance_integrated(X, y, model, method='MDI', X_val=None, y_val=None, n_repeats=10, seed=42):
    """
    Computes the integrated feature importance of a sckit-learn type model.

    :param X: dataframe - features to train the model on
    :param y: array/ Serie
    :param model: scikit-learn type model. Model must include a feature_importances_ field !
    :param X_val: dataframe - Validation data
    :param y_val: dataframe - Validation target
    :param method: dataframe - Feature importance method ("MDA" or "MDI")

    :return: feature_importances : dataframe with features sorted by their importances
    """

    feature_importances = None

    if method == 'MDI':
        model.fit(X, y)
        feature_importances = DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_
        })
        feature_importances.sort_values(by='importance', ascending=False, inplace=True)

    elif method == 'MDA':
        model.fit(X, y)
        result = permutation_importance(model, X_val, y_val, n_repeats=n_repeats, random_state=seed)
        feature_importances = DataFrame(result.importances.T, columns=X.columns)

    return feature_importances


def feature_importance_performance(X, y, model, cv=False, scoring_func=mean_squared_error, verbose=0):
    performances = []
    if not cv:
        features = list(X.columns)
        model.fit(X, y)
        score = scoring_func(y, model.predict(X))
        performances.append({
            "features": features,
            "score": score
        })

        while len(features) > 0:
            if verbose > 0:
                print("Training for ", len(features) - 1, "features...")
            worst_feature, score = _worst_performing_feature(X, y, features, model, scoring_func=scoring_func)
            features = list(set(features) - set(worst_feature))
            if verbose > 0:
                print(worst_feature, "has been eliminated !")
            performances.append({
                'features': features,
                'score': score
            })

    return performances


def recursive_feature_elimination(X, y, dates, model, n_features, n_features_min,  show=False,
                                  scoring_func=mean_squared_error, evaluate=True):
    print("Loading models...")
    models = _get_models_rfe(model, n_features=n_features, n_features_min=n_features_min)
    results, names = list(), list()
    cv_list = _yearly_cv_separation(X, y, dates)

    for name, model in models.items():
        if evaluate:
            print("Model", name, "is being evaluated...")
            scores = _evaluate_model_rfe(model, cv_list, scoring_func=scoring_func)
            print('>{}, {} ({})'.format(name, mean(scores), std(scores)))
            results.append(scores)
        model.fit(X, y)
        names.append(name)

    if show:
        # plot model performance for comparison
        plt.boxplot(results, labels=names, showmeans=True)
        plt.show()

    return models, results


def _get_models_rfe(model, n_features, n_features_min):
    models = dict()
    for i in range(n_features-1, n_features_min, -1):
        rfe = RFE(estimator=clone(model), n_features_to_select=i, verbose=1)
        model = clone(model)
        models[str(i)] = Pipeline(steps=[('s', rfe), ('m', model)])
    return models


def _evaluate_model_rfe(model, cv_list, scoring_func=mean_squared_error):
    cv_res = []
    for (X_train, y_train, X_test, y_test) in cv_list:
        model_copy = clone(model)
        model_copy.fit(X_train, y_train)
        cv_res.append(scoring_func(y_test, model_copy.predict(X_test)))

    return cv_res


def _yearly_cv_separation(X, y, dates):
    df = X.copy()
    df['Date'] = pd.to_datetime(dates)
    df['SpotPrice'] = y

    assert ((df.Date.dt.year.max() == 2020) and (df.Date.dt.year.min() == 2018)), "Must include years 2018, 2019, 2020"

    cv_list = [(df.loc[~(df.Date.dt.year == year), X.columns],
                df.loc[~(df.Date.dt.year == year), 'SpotPrice'],
                df.loc[df.Date.dt.year == year, X.columns],
                df.loc[df.Date.dt.year == year, 'SpotPrice']
                )
               for year in [2018, 2019, 2020]]

    return cv_list


def _worst_performing_feature(X, y, features, model, scoring_func=mean_squared_error):
    dic_performance = {}
    for feature in features:
        new_features = list(set(features) - set(feature))
        new_X = X[new_features].copy()
        model.fit(new_X, y)
        score = scoring_func(y, model.predict(new_X))
        dic_performance[feature] = score

    worst_feature = min(dic_performance, key=dic_performance.get)
    best_score = min(dic_performance.values())
    return worst_feature, best_score
