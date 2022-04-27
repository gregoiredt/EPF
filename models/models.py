from sklearn.base import clone


class HourlyModel:
    """
    Model encompassing every hourly model for price prediction
    """

    def __init__(self, estimator=None, dic_models=None):
        assert not (estimator is None) or not (dic_models is None), "An estimator or list of estimators must be given"
        # FTM : Only works for sklearn type estimator if lis_models is not given
        self.list_models = dic_models if not (dic_models is None) else {i: clone(estimator) for i in range(0, 24)}

    def fit(self, X, y):
        for hour, model in self.list_models.items():
            model.fit(X[X.Hour == hour], y[X.Hour == hour])


