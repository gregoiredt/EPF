from pandas import read_csv
import numpy as np
#from ..cqr import torch_models
from functools import partial
import os
from ..cqr import tune_params_cv
from ..nonconformist.cp import IcpRegressor
from ..nonconformist.base import RegressorAdapter
from skgarden import RandomForestQuantileRegressor
from sklearn.base import clone
from .models import AsinhScaler, AsinhMedianScaler, MedianScaler
from loguru import logger
import torch

import pdb

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"




def compute_coverage_len(y_test, y_lower, y_upper):
    """ Compute average coverage and length of prediction intervals

    Parameters
    ----------

    y_test : numpy array, true labels (n)
    y_lower : numpy array, estimated lower bound for the labels (n)
    y_upper : numpy array, estimated upper bound for the labels (n)

    Returns
    -------

    coverage : float, average coverage
    avg_length : float, average length

    """
    in_the_range = np.sum((y_test >= y_lower) & (y_test <= y_upper))
    coverage = in_the_range / len(y_test) * 100
    avg_length = np.mean(abs(y_upper - y_lower))
    return coverage, avg_length


def run_icp(nc, X_train, y_train, X_test, idx_train, idx_cal, significance, condition=None, y_test=None,
            method='normal'):
    """ Run split conformal method

    Parameters
    ----------

    nc : class of nonconformist object
    X_train : numpy array, training features (n1Xp)
    y_train : numpy array, training labels (n1)
    X_test : numpy array, testing features (n2Xp)
    idx_train : numpy array, indices of proper training set examples
    idx_cal : numpy array, indices of calibration set examples
    significance : float, significance level (e.g. 0.1)
    condition : function, mapping feature vector to group id

    Returns
    -------

    y_lower : numpy array, estimated lower bound for the labels (n2)
    y_upper : numpy array, estimated upper bound for the labels (n2)

    """
    icp = IcpRegressor(nc, condition=condition)

    # Fit the ICP using the proper training set
    icp.fit(X_train[idx_train, :], y_train[idx_train])

    # Calibrate the ICP using the calibration set
    icp.calibrate(X_train[idx_cal, :], y_train[idx_cal])

    # Produce predictions for the test set, with confidence 90%
    predictions = icp.predict(X_test, significance=significance, y_test=y_test)

    if method == 'no_conformal':
        predictions_no_conformal = icp.predict(X_test, significance=significance, y_test=y_test, no_conformal=True)

        return predictions, predictions_no_conformal

    return predictions


# TODO : Implement for mean predictions
def run_ojacknife(nc, X_train, y_train, X_test, idx_train, idx_cal, significance, condition=None, y_test=None,
                  method='CQR', print_=False):
    """ Run Online Jacknife prediction

    Parameters
    ----------

    nc : class of nonconformist object
    X_train : numpy array, training features (n1Xp)
    y_train : numpy array, training labels (n1)
    X_test : numpy array, testing features (n2Xp)
    idx_train : numpy array, indices of proper training set examples
    idx_cal : numpy array, indices of calibration set examples
    significance : float, significance level (e.g. 0.1)
    condition : function, mapping feature vector to group id

    Returns
    -------

    y_lower : numpy array, estimated lower bound for the labels (n2)
    y_upper : numpy array, estimated upper bound for the labels (n2)

    """
    n_train, n_cal, n_test = idx_train.shape[0], idx_cal.shape[0], X_test.shape[0]
    saved_cal_scores = np.zeros((n_cal, len(significance)))
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    idx_test = np.arange(len(X_test)) + np.max(idx_cal) + 1
    first_idx_test = np.min(idx_test)
    idx_cal_test = np.concatenate([idx_cal, idx_test])

    predictions = np.zeros((X_test.shape[0], 2, len(significance))) if method=='CQR' \
        else np.zeros((X_test.shape[0], 2, len(significance), len(nc.gammas)))
    for i, idx in enumerate(idx_cal_test):
        icp = IcpRegressor(nc, condition=condition)
        nidx_train = np.concatenate([idx_train[i:], idx_cal_test[max(0, i - n_train):i]])

        X_copy, y_copy = X.copy(), y.copy()

        if print_:
            logger.info(f' -------  Starting {i}-th step (out of {len(idx_cal_test[:-1])})... -------- ')
            logger.info(f'Nidx_train : {nidx_train} ')
            logger.info(f'idx : {idx}')

        # Fit the ICP using the proper training set
        icp.fit(X_copy[nidx_train, :], y_copy[nidx_train])
        icp.categories = np.array([0])

        # Predict on test_set
        if idx >= first_idx_test:
            if print_:
                logger.info('testing...')
            icp.cal_scores = {0: np.sort(saved_cal_scores, 0)[::-1]}
            predictions[i - n_cal] = icp.predict(X_copy[[idx]], significance=significance,
                                                     y_test=y_copy[[idx]]).squeeze(0)
            if print_:
                logger.info(f'adaptative significance : {nc.significance_t}')
                logger.info(f'index of test {i  - n_cal}')

        # Calibrate the ICP using the calibration set
        icp.calibrate(X_copy[[idx], :], y_copy[[idx]])
        saved_cal_scores[i % n_cal] = icp.cal_scores[0]  # only one cal_score

        if print_:
            logger.info(f'Calibration scores : {icp.cal_scores[0]}')

    return predictions


def _deprecated_run_ojacknife(nc, X_train, y_train, X_test, idx_train, idx_cal, significance, condition=None, y_test=None,
                  method='CQR', print_=False):
    """ Run Online Jacknife prediction

    Parameters
    ----------

    nc : class of nonconformist object
    X_train : numpy array, training features (n1Xp)
    y_train : numpy array, training labels (n1)
    X_test : numpy array, testing features (n2Xp)
    idx_train : numpy array, indices of proper training set examples
    idx_cal : numpy array, indices of calibration set examples
    significance : float, significance level (e.g. 0.1)
    condition : function, mapping feature vector to group id

    Returns
    -------

    y_lower : numpy array, estimated lower bound for the labels (n2)
    y_upper : numpy array, estimated upper bound for the labels (n2)

    """
    n_train, n_cal, n_test = idx_train.shape[0], idx_cal.shape[0], X_test.shape[0]
    saved_cal_scores = np.zeros((n_cal, len(significance)))
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    idx_test = np.arange(len(X_test)) + np.max(idx_cal) + 1
    first_idx_test = np.min(idx_test)
    idx_cal_test = np.concatenate([idx_cal, idx_test])

    predictions = np.zeros((X_test.shape[0], 2, len(significance))) if method=='CQR' \
        else np.zeros((X_test.shape[0], 2, len(significance), len(nc.gammas)))
    for i, idx in enumerate(idx_cal_test[:-1]):
        icp = IcpRegressor(nc, condition=condition)
        nidx_train = np.concatenate([idx_train[i:], idx_cal_test[max(0, i - n_train):i]])

        X_copy, y_copy = X.copy(), y.copy()

        if print_:
            logger.info(f' -------  Starting {i}-th step (out of {len(idx_cal_test[:-1])})... -------- ')
            logger.info(f'Nidx_train : {nidx_train} ')
            logger.info(f'idx : {idx}')

        # Fit the ICP using the proper training set
        icp.fit(X_copy[nidx_train, :], y_copy[nidx_train])

        # Calibrate the ICP using the calibration set
        icp.calibrate(X_copy[[idx], :], y_copy[[idx]])
        saved_cal_scores[i % n_cal] = icp.cal_scores[0]  # only one cal_score

        # Predict on test_set
        if idx + 1 >= first_idx_test:
            icp.cal_scores = {0: np.sort(saved_cal_scores, 0)[::-1]}
            predictions[i + 1 - n_cal] = icp.predict(X_copy[[idx + 1]], significance=significance,
                                                     y_test=y_copy[[idx + 1]]).squeeze(0)
            if print_:
                logger.info(f'adaptative significance : {nc.significance_t}')

    return predictions


def run_ojacknifeplus(nc, X_train, y_train, X_test, idx_train, idx_cal, significance, condition=None, y_test=None,
                      method='CQR', print_=False):
    """ Run Online Jacknife+ prediction

    Parameters
    ----------

    nc : class of nonconformist object
    X_train : numpy array, training features (n1Xp)
    y_train : numpy array, training labels (n1)
    X_test : numpy array, testing features (n2Xp)
    idx_train : numpy array, indices of proper training set examples
    idx_cal : numpy array, indices of calibration set examples
    significance : float, significance level (e.g. 0.1)
    condition : function, mapping feature vector to group id

    Returns
    -------

    y_lower : numpy array, estimated lower bound for the labels (n2)
    y_upper : numpy array, estimated upper bound for the labels (n2)

    """
    n_train, n_cal, n_test = idx_train.shape[0], idx_cal.shape[0], X_test.shape[0]
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    idx_test = np.arange(len(X_test)) + np.max(idx_cal) + 1
    idx_cal_test = np.concatenate([idx_cal, idx_test])

    prediction_scores = np.zeros((n_cal, n_test, 2, len(significance)))
    for i, idx in enumerate(idx_cal_test[:-1]):

        icp = IcpRegressor(nc, condition=condition)
        nidx_train = np.concatenate([idx_train[i:], idx_cal_test[max(0, i - n_train):i]])

        # Fit the ICP using the proper training set
        X_to_fit, y_to_fit = X[nidx_train, :].copy(), y[nidx_train].copy()
        X_to_cal, y_to_cal = X[[idx], :].copy(), y[[idx]].copy()
        X_to_test, y_to_test = X_test.copy(), y_test.copy()

        if print_:
            logger.info(f'-------  Starting {i}-th step (out of {len(idx_cal_test[:-1])})...------- ')
            logger.info(f'Nidx_train : {nidx_train} ')
            logger.info(f'idx : {idx}')

        icp.fit(X_to_fit, y_to_fit)

        # Calibrate the ICP using the calibration set
        icp.calibrate(X_to_cal, y_to_cal)
        cal_score = icp.cal_scores[0]  # shape (1, n_significance)

        # Predict on test set (without conformalisation)
        predictions = icp.predict(X_to_test, significance=significance, y_test=y_to_test,
                                  no_conformal=True)  # (n_test, 2, n_significance)

        if print_:
            logger.info(f'N_predictions : {predictions.shape}')
            logger.info(f'predictions : {predictions}')
        for j, alpha in enumerate(significance):
            if predictions.ndim == 2:
                # TODO : To implement
                pass
            elif predictions.ndim > 2:
                if print_:
                    logger.info(f"alpha {j}-th : {alpha}")
                    logger.info(f"Indices : {(i % n_cal , max(i - n_cal, 0), j)}")
                    logger.info(f'n preds : {len(predictions[max(i - n_cal, 0):, 0, j])}')
                    logger.info(f'preds : {predictions[max(i - n_cal, 0):, 0, j]}')
                    logger.info(f'n preds : {predictions[:, 0, j]}')
                # shape (722, 2)
                prediction_scores[i % n_cal, max(i - n_cal, 0):, 0, j] = predictions[max(i - n_cal, 0):, 0, j] - cal_score[0, j]
                prediction_scores[i % n_cal, max(i - n_cal, 0):, 1, -(j + 1)] = predictions[max(i - n_cal, 0):, 1, -(j + 1)] + cal_score[0, j]


    # Retrieve predictions:
    final_predictions = np.zeros((n_test, 2, len(significance)))
    for j, alpha in enumerate(significance):
        nc_scores_low = prediction_scores[:, :, 0, j]  # ncal, n_test
        nc_scores_high = prediction_scores[:, :, 1, j]

        index = int(np.ceil((1 - alpha) * (nc_scores_low.shape[0] + 1))) - 1
        index = min(max(index, 0), nc_scores_low.shape[0] - 1)  # Same index or high or low

        pred_low = np.sort(nc_scores_low, 0)[::-1][index, :]
        pred_high = np.sort(nc_scores_high, 0)[index, :]

        final_predictions[:, :, j] = np.vstack([pred_low, pred_high]).T

    return final_predictions


def run_icp_sep(nc, X_train, y_train, X_test, idx_train, idx_cal, significance, condition):
    """ Run split conformal method, train a seperate regressor for each group

    Parameters
    ----------

    nc : class of nonconformist object
    X_train : numpy array, training features (n1Xp)
    y_train : numpy array, training labels (n1)
    X_test : numpy array, testing features (n2Xp)
    idx_train : numpy array, indices of proper training set examples
    idx_cal : numpy array, indices of calibration set examples
    significance : float, significance level (e.g. 0.1)
    condition : function, mapping a feature vector to group id

    Returns
    -------

    y_lower : numpy array, estimated lower bound for the labels (n2)
    y_upper : numpy array, estimated upper bound for the labels (n2)

    """

    X_proper_train = X_train[idx_train, :]
    y_proper_train = y_train[idx_train]
    X_calibration = X_train[idx_cal, :]
    y_calibration = y_train[idx_cal]

    category_map_proper_train = np.array(
        [condition((X_proper_train[i, :], y_proper_train[i])) for i in range(y_proper_train.size)])
    category_map_calibration = np.array(
        [condition((X_calibration[i, :], y_calibration[i])) for i in range(y_calibration.size)])
    category_map_test = np.array([condition((X_test[i, :], None)) for i in range(X_test.shape[0])])

    categories = np.unique(category_map_proper_train)

    y_lower = np.zeros(X_test.shape[0])
    y_upper = np.zeros(X_test.shape[0])

    cnt = 0

    for cond in categories:
        icp = IcpRegressor(nc[cnt])

        idx_proper_train_group = category_map_proper_train == cond
        # Fit the ICP using the proper training set
        icp.fit(X_proper_train[idx_proper_train_group, :], y_proper_train[idx_proper_train_group])

        idx_calibration_group = category_map_calibration == cond
        # Calibrate the ICP using the calibration set
        icp.calibrate(X_calibration[idx_calibration_group, :], y_calibration[idx_calibration_group])

        idx_test_group = category_map_test == cond
        # Produce predictions for the test set, with confidence 90%
        predictions = icp.predict(X_test[idx_test_group, :], significance=significance)

        y_lower[idx_test_group] = predictions[:, 0]
        y_upper[idx_test_group] = predictions[:, 1]

        cnt = cnt + 1

    return y_lower, y_upper


def compute_coverage(y_test, y_lower, y_upper, significance, name=""):
    """ Compute average coverage and length, and print results

    Parameters
    ----------

    y_test : numpy array, true labels (n)
    y_lower : numpy array, estimated lower bound for the labels (n)
    y_upper : numpy array, estimated upper bound for the labels (n)
    significance : float, desired significance level
    name : string, optional output string (e.g. the method name)

    Returns
    -------

    coverage : float, average coverage
    avg_length : float, average length

    """
    in_the_range = np.sum((y_test >= y_lower) & (y_test <= y_upper))
    coverage = in_the_range / len(y_test) * 100
    print("%s: Percentage in the range (expecting %.2f): %f" % (name, 100 - significance * 100, coverage))
    sys.stdout.flush()

    avg_length = abs(np.mean(y_lower - y_upper))
    print("%s: Average length: %f" % (name, avg_length))
    sys.stdout.flush()
    return coverage, avg_length


def compute_coverage_per_sample(y_test, y_lower, y_upper, significance, name="", x_test=None, condition=None):
    """ Compute average coverage and length, and print results

    Parameters
    ----------

    y_test : numpy array, true labels (n)
    y_lower : numpy array, estimated lower bound for the labels (n)
    y_upper : numpy array, estimated upper bound for the labels (n)
    significance : float, desired significance level
    name : string, optional output string (e.g. the method name)
    x_test : numpy array, test features
    condition : function, mapping a feature vector to group id

    Returns
    -------

    coverage : float, average coverage
    avg_length : float, average length

    """

    if condition is not None:

        category_map = np.array([condition((x_test[i, :], y_test[i])) for i in range(y_test.size)])
        categories = np.unique(category_map)

        coverage = np.empty(len(categories), dtype=np.object)
        length = np.empty(len(categories), dtype=np.object)

        cnt = 0

        for cond in categories:
            idx = category_map == cond

            coverage[cnt] = (y_test[idx] >= y_lower[idx]) & (y_test[idx] <= y_upper[idx])

            coverage_avg = np.sum(coverage[cnt]) / len(y_test[idx]) * 100
            print("%s: Group %d : Percentage in the range (expecting %.2f): %f" % (
                name, cond, 100 - significance * 100, coverage_avg))
            sys.stdout.flush()

            length[cnt] = abs(y_upper[idx] - y_lower[idx])
            print("%s: Group %d : Average length: %f" % (name, cond, np.mean(length[cnt])))
            sys.stdout.flush()
            cnt = cnt + 1

    else:

        coverage = (y_test >= y_lower) & (y_test <= y_upper)
        coverage_avg = np.sum(coverage) / len(y_test) * 100
        print("%s: Percentage in the range (expecting %.2f): %f" % (name, 100 - significance * 100, coverage_avg))
        sys.stdout.flush()

        length = abs(y_upper - y_lower)
        print("%s: Average length: %f" % (name, np.mean(length)))
        sys.stdout.flush()

    return coverage, length


def plot_func_data(y_test, y_lower, y_upper, name=""):
    """ Plot the test labels along with the constructed prediction band

    Parameters
    ----------

    y_test : numpy array, true labels (n)
    y_lower : numpy array, estimated lower bound for the labels (n)
    y_upper : numpy array, estimated upper bound for the labels (n)
    name : string, optional output string (e.g. the method name)

    """

    # allowed to import graphics
    import matplotlib.pyplot as plt

    interval = y_upper - y_lower
    sort_ind = np.argsort(interval)
    y_test_sorted = y_test[sort_ind]
    upper_sorted = y_upper[sort_ind]
    lower_sorted = y_lower[sort_ind]
    mean = (upper_sorted + lower_sorted) / 2

    # Center such that the mean of the prediction interval is at 0.0
    y_test_sorted -= mean
    upper_sorted -= mean
    lower_sorted -= mean

    plt.plot(y_test_sorted, "ro")
    plt.fill_between(
        np.arange(len(upper_sorted)), lower_sorted, upper_sorted, alpha=0.2, color="r",
        label="Pred. interval")
    plt.xlabel("Ordered samples")
    plt.ylabel("Values and prediction intervals")

    plt.title(name)
    plt.show()

    interval = y_upper - y_lower
    sort_ind = np.argsort(y_test)
    y_test_sorted = y_test[sort_ind]
    upper_sorted = y_upper[sort_ind]
    lower_sorted = y_lower[sort_ind]

    plt.plot(y_test_sorted, "ro")
    plt.fill_between(
        np.arange(len(upper_sorted)), lower_sorted, upper_sorted, alpha=0.2, color="r",
        label="Pred. interval")
    plt.xlabel("Ordered samples by response")
    plt.ylabel("Values and prediction intervals")

    plt.title(name)
    plt.show()


###############################################################################
# Deep conditional mean regression
# Minimizing MSE loss
###############################################################################

class MSENet_RegressorAdapter(RegressorAdapter):
    """ Conditional mean estimator, formulated as neural net
    """

    def __init__(self,
                 model,
                 fit_params=None,
                 in_shape=1,
                 hidden_size=1,
                 learn_func=torch.optim.Adam,
                 epochs=1000,
                 batch_size=10,
                 dropout=0.1,
                 lr=0.01,
                 wd=1e-6,
                 test_ratio=0.2,
                 random_state=0):
        """ Initialization

        Parameters
        ----------
        model : unused parameter (for compatibility with nc class)
        fit_params : unused parameter (for compatibility with nc class)
        in_shape : integer, input signal dimension
        hidden_size : integer, hidden layer dimension
        learn_func : class of Pytorch's SGD optimizer
        epochs : integer, maximal number of epochs
        batch_size : integer, mini-batch size for SGD
        dropout : float, dropout rate
        lr : float, learning rate for SGD
        wd : float, weight decay
        test_ratio : float, ratio of held-out data, used in cross-validation
        random_state : integer, seed for splitting the data in cross-validation

        """
        super(MSENet_RegressorAdapter, self).__init__(model, fit_params)
        # Instantiate model
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.lr = lr
        self.wd = wd
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.model = torch_models.mse_model(in_shape=in_shape, hidden_size=hidden_size, dropout=dropout)
        self.loss_func = torch.nn.MSELoss()
        self.learner = torch_models.LearnerOptimized(self.model,
                                                     partial(learn_func, lr=lr, weight_decay=wd),
                                                     self.loss_func,
                                                     device=device,
                                                     test_ratio=self.test_ratio,
                                                     random_state=self.random_state)

    def fit(self, x, y):
        """ Fit the model to data

        Parameters
        ----------

        x : numpy array of training features (nXp)
        y : numpy array of training labels (n)

        """
        self.learner.fit(x, y, self.epochs, batch_size=self.batch_size)

    def predict(self, x):
        """ Estimate the label given the features

        Parameters
        ----------
        x : numpy array of training features (nXp)

        Returns
        -------
        ret_val : numpy array of predicted labels (n)

        """
        return self.learner.predict(x)


###############################################################################
# Deep neural network for conditional quantile regression
# Minimizing pinball loss
###############################################################################

class AllQNet_RegressorAdapter(RegressorAdapter):
    """ Conditional quantile estimator, formulated as neural net
    """

    def __init__(self,
                 model,
                 fit_params=None,
                 in_shape=1,
                 hidden_size=1,
                 quantiles=[.05, .95],
                 learn_func=torch.optim.Adam,
                 epochs=1000,
                 batch_size=10,
                 dropout=0.1,
                 lr=0.01,
                 wd=1e-6,
                 test_ratio=0.2,
                 random_state=0,
                 use_rearrangement=False):
        """ Initialization

        Parameters
        ----------
        model : None, unused parameter (for compatibility with nc class)
        fit_params : None, unused parameter (for compatibility with nc class)
        in_shape : integer, input signal dimension
        hidden_size : integer, hidden layer dimension
        quantiles : numpy array, low and high quantile levels in range (0,1)
        learn_func : class of Pytorch's SGD optimizer
        epochs : integer, maximal number of epochs
        batch_size : integer, mini-batch size for SGD
        dropout : float, dropout rate
        lr : float, learning rate for SGD
        wd : float, weight decay
        test_ratio : float, ratio of held-out data, used in cross-validation
        random_state : integer, seed for splitting the data in cross-validation
        use_rearrangement : boolean, use the rearrangement algorithm (True)
                            of not (False). See reference [1].

        References
        ----------
        .. [1]  Chernozhukov, Victor, Iván Fernández‐Val, and Alfred Galichon.
                "Quantile and probability curves without crossing."
                Econometrica 78.3 (2010): 1093-1125.

        """
        super(AllQNet_RegressorAdapter, self).__init__(model, fit_params)
        # Instantiate model
        self.quantiles = quantiles
        if use_rearrangement:
            self.all_quantiles = torch.from_numpy(np.linspace(0.01, 0.99, 99)).float()
        else:
            self.all_quantiles = self.quantiles
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.lr = lr
        self.wd = wd
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.model = torch_models.all_q_model(quantiles=self.all_quantiles,
                                              in_shape=in_shape,
                                              hidden_size=hidden_size,
                                              dropout=dropout)
        self.loss_func = torch_models.AllQuantileLoss(self.all_quantiles)
        self.learner = torch_models.LearnerOptimizedCrossing(self.model,
                                                             partial(learn_func, lr=lr, weight_decay=wd),
                                                             self.loss_func,
                                                             device=device,
                                                             test_ratio=self.test_ratio,
                                                             random_state=self.random_state,
                                                             qlow=self.quantiles[0],
                                                             qhigh=self.quantiles[1],
                                                             use_rearrangement=use_rearrangement)

    def fit(self, x, y):
        """ Fit the model to data

        Parameters
        ----------

        x : numpy array of training features (nXp)
        y : numpy array of training labels (n)

        """
        self.learner.fit(x, y, self.epochs, self.batch_size)

    def predict(self, x):
        """ Estimate the conditional low and high quantiles given the features

        Parameters
        ----------
        x : numpy array of training features (nXp)

        Returns
        -------
        ret_val : numpy array of estimated conditional quantiles (nX2)

        """
        return self.learner.predict(x)


###############################################################################
# Quantile random forests model
###############################################################################

class QuantileForestRegressorAdapter(RegressorAdapter):
    """ Conditional quantile estimator, defined as quantile random forests (QRF)

    References
    ----------
    .. [1]  Meinshausen, Nicolai. "Quantile regression forests."
            Journal of Machine Learning Research 7.Jun (2006): 983-999.

    """

    def __init__(self,
                 model,
                 fit_params=None,
                 quantiles=None,
                 params=None,
                 preprocessing=True):
        """ Initialization

        Parameters
        ----------
        model : None, unused parameter (for compatibility with nc class)
        fit_params : None, unused parameter (for compatibility with nc class)
        quantiles : numpy array, low and high quantile levels in range (0,100)
        params : dictionary of parameters
                params["random_state"] : integer, seed for splitting the data
                                         in cross-validation. Also used as the
                                         seed in quantile random forests (QRF)
                params["min_samples_leaf"] : integer, parameter of QRF
                params["n_estimators"] : integer, parameter of QRF
                params["max_features"] : integer, parameter of QRF
                params["CV"] : boolean, use cross-validation (True) or
                               not (False) to tune the two QRF quantile levels
                               to obtain the desired coverage
                params["test_ratio"] : float, ratio of held-out data, used
                                       in cross-validation
                params["coverage_factor"] : float, to avoid too conservative
                                            estimation of the prediction band,
                                            when tuning the two QRF quantile
                                            levels in cross-validation one may
                                            ask for prediction intervals with
                                            reduced average coverage, equal to
                                            coverage_factor*(q_high - q_low).
                params["range_vals"] : float, determines the lowest and highest
                                       quantile level parameters when tuning
                                       the quanitle levels bt cross-validation.
                                       The smallest value is equal to
                                       quantiles[0] - range_vals.
                                       Similarly, the largest is equal to
                                       quantiles[1] + range_vals.
                params["num_vals"] : integer, when tuning QRF's quantile
                                     parameters, sweep over a grid of length
                                     num_vals.

        """
        super(QuantileForestRegressorAdapter, self).__init__(model, fit_params)
        # Instantiate model
        self.quantiles = quantiles
        self.cv_quantiles = self.quantiles
        self.params = params
        self.preprocessing = preprocessing
        self.rfqr = RandomForestQuantileRegressor(random_state=params["random_state"],
                                                  min_samples_leaf=params["min_samples_leaf"],
                                                  n_estimators=params["n_estimators"],
                                                  max_features=params["max_features"])

    def fit(self, x, y):
        """ Fit the model to data

        Parameters
        ----------

        x : numpy array of training features (nXp)
        y : numpy array of training labels (n)

        """
        self.cv_quantiles = []

        if self.preprocessing:
            self.y_scaler = AsinhMedianScaler()
            self.x_scaler = MedianScaler()

            self.y_scaler.fit(y)
            self.x_scaler.fit(x)

            x, y = self.x_scaler.transform(x.copy()), self.y_scaler.transform(y.copy())

        for quants in self.quantiles:
            q_low, q_high = quants[0], quants[1]
            if self.params["CV"]:
                target_coverage = q_high - q_low
                coverage_factor = self.params["coverage_factor"]
                range_vals = self.params["range_vals"]
                num_vals = self.params["num_vals"]
                grid_q_low = np.linspace(q_low, q_low + range_vals, num_vals).reshape(-1, 1)
                grid_q_high = np.linspace(q_high, q_high - range_vals, num_vals).reshape(-1, 1)
                grid_q = np.concatenate((grid_q_low, grid_q_high), 1)

                self.cv_quantiles.append(tune_params_cv.CV_quntiles_rf(self.params,
                                                                       x,
                                                                       y,
                                                                       target_coverage,
                                                                       grid_q,
                                                                       self.params["test_ratio"],
                                                                       self.params["random_state"],
                                                                       coverage_factor))
            else:
                self.cv_quantiles.append(np.array([q_low, q_high]))

        self.rfqr.fit(x, y)

    def predict(self, x):
        """ Estimate the conditional low and high quantiles given the features

        Parameters
        ----------
        x : numpy array of training features (nXp)

        Returns
        -------
        ret_val : numpy array of estimated conditional quantiles (n, 2 * n_intervals)

        """
        if self.preprocessing:
            x = self.x_scaler.transform(x.copy())

        ret_val = np.zeros((x.shape[0], 2 * len(self.quantiles)))
        for i, cv_quantile in enumerate(self.cv_quantiles):
            lower = self.rfqr.predict(x, quantile=cv_quantile[0])
            upper = self.rfqr.predict(x, quantile=cv_quantile[1])

            if self.preprocessing:
                lower = self.y_scaler.inverse_transform(lower)
                upper = self.y_scaler.inverse_transform(upper)

            ret_val[:, i] = lower
            ret_val[:, -(i + 1)] = upper

        return ret_val


###############################################################################
# Adapative Quantile random forests model
###############################################################################


class AdaptativeQuantileForestRegressorAdapter(RegressorAdapter):
    """ Adaptative quantile random forest

    References
    ----------
    .. [1]  Meinshausen, Nicolai. "Quantile regression forests."
            Journal of Machine Learning Research 7.Jun (2006): 983-999.

    """

    def __init__(self,
                 model,
                 gamma,
                 fit_params=None,
                 quantiles=None,
                 adaptative_quantiles=None,
                 params=None,
                 preprocessing=True):
        """ Initialization

        Parameters
        ----------
        model : None, unused parameter (for compatibility with nc class)
        gamma : float, adaptation parameter (must be positive)
        fit_params : None, unused parameter (for compatibility with nc class)
        quantiles : numpy array, low and high quantile levels in range (0,100)
        adaptative_quantiles : numpy array, low and high quantile levels in range (0,100), supposed to be adapted
        params : dictionary of parameters
                params["random_state"] : integer, seed for splitting the data
                                         in cross-validation. Also used as the
                                         seed in quantile random forests (QRF)
                params["min_samples_leaf"] : integer, parameter of QRF
                params["n_estimators"] : integer, parameter of QRF
                params["max_features"] : integer, parameter of QRF
                params["CV"] : boolean, use cross-validation (True) or
                               not (False) to tune the two QRF quantile levels
                               to obtain the desired coverage
                params["test_ratio"] : float, ratio of held-out data, used
                                       in cross-validation
                params["coverage_factor"] : float, to avoid too conservative
                                            estimation of the prediction band,
                                            when tuning the two QRF quantile
                                            levels in cross-validation one may
                                            ask for prediction intervals with
                                            reduced average coverage, equal to
                                            coverage_factor*(q_high - q_low).
                params["range_vals"] : float, determines the lowest and highest
                                       quantile level parameters when tuning
                                       the quanitle levels bt cross-validation.
                                       The smallest value is equal to
                                       quantiles[0] - range_vals.
                                       Similarly, the largest is equal to
                                       quantiles[1] + range_vals.
                params["num_vals"] : integer, when tuning QRF's quantile
                                     parameters, sweep over a grid of length
                                     num_vals.

        """
        super(AdaptativeQuantileForestRegressorAdapter, self).__init__(model, fit_params)
        # Instantiate model
        self.quantiles = np.array(quantiles)
        self.adaptative_quantiles = adaptative_quantiles
        self.params = params
        self.gamma = gamma
        self.preprocessing = preprocessing
        self.rfqr = RandomForestQuantileRegressor(random_state=params["random_state"],
                                                  min_samples_leaf=params["min_samples_leaf"],
                                                  n_estimators=params["n_estimators"],
                                                  max_features=params["max_features"])

    def fit(self, x, y):
        """ Fit the model to data

        Parameters
        ----------

        x : numpy array of training features (nXp)
        y : numpy array of training labels (n)

        """

        if self.preprocessing:
            self.y_scaler = AsinhMedianScaler()
            self.x_scaler = MedianScaler()

            self.y_scaler.fit(y)
            self.x_scaler.fit(x)

            self.rfqr.fit(self.x_scaler.transform(x), self.y_scaler.transform(y))

        else:
            self.rfqr.fit(x, y)

    def update_quantiles(self, y_true, y_pred):
        """Updates the quantile given the formula : ...

        Parameters
        ----------
        y_true: numpy array of size 1
        y_pred: numpy array of size (2 * n_interval)
        :return:
        """

        n_interval = len(y_pred) // 2
        y_true = np.repeat(y_true, 2 * n_interval)

        err = np.zeros(2 * n_interval)
        err[:n_interval] = (y_true[:n_interval] < y_pred[:n_interval]).astype(int)
        err[n_interval:] = (y_true[n_interval:] > y_pred[n_interval:]).astype(int)

        new_quantiles = np.zeros_like(self.adaptative_quantiles)

        for i in range(n_interval):
            new_quantiles[i, 0] = 100 * ((self.adaptative_quantiles[i, 0] / 100) + self.gamma * (self.quantiles[i, 0] / 100 - err[i]))
            new_quantiles[i, 1] = 100 * ((self.adaptative_quantiles[i, 1] / 100) + self.gamma * (err[-(i + 1)] - (1 - (self.quantiles[i, 1]) / 100)))

        # ceil
        new_quantiles[np.where(new_quantiles > 99.9)[0]] = 99.9
        new_quantiles[np.where(new_quantiles < 0.1)[0]] = 0.1
        self.adaptative_quantiles = new_quantiles

        return new_quantiles

    def predict(self, x):
        """ Estimate the conditional low and high quantiles given the features

        Parameters
        ----------
        x : numpy array of training features (nXp)

        Returns
        -------
        ret_val : numpy array of estimated conditional quantiles (n, 2 * n_intervals)

        """
        if self.preprocessing:
            x = self.x_scaler.transform(x.copy())

        ret_val = np.zeros((x.shape[0], 2 * len(self.quantiles)))
        for i, aquantile in enumerate(self.adaptative_quantiles):
            lower = self.rfqr.predict(x, quantile=aquantile[0])
            upper = self.rfqr.predict(x, quantile=aquantile[1])

            if self.preprocessing:
                lower = self.y_scaler.inverse_transform(lower)
                upper = self.y_scaler.inverse_transform(upper)

            ret_val[:, i] = lower
            ret_val[:, -(i + 1)] = upper

        return ret_val


###############################################################################
# Custom Sklearn Regressor Adapater
###############################################################################
class CustomSklearnRegressorAdapter(RegressorAdapter):
    """ Conditional quantile estimator based on a sklearn model. This type of models usually have to be trained for every
    quantiles. If a mean regressor is provided, the regressor perform classical mean regression.
    References
    ----------
    """

    def __init__(self,
                 model,
                 fit_params=None,
                 quantiles=None,
                 params=None):
        """ Initialization

        Parameters
        ----------
        model : Scikit-learn model
        fit_params : None, used for compatibility with nc class
        quantiles : numpy array, low and high quantile levels in range (0,100)
        """
        super(CustomSklearnRegressorAdapter, self).__init__(model, fit_params)
        # Instantiate model
        self.quantiles = np.sort(quantiles)
        self.model = model
        self.is_classical_regressor = self.quantiles is None
        self.location_file = params['location_file']
        self.params = params
        self.dic = {}

    def fit(self, X, y, *args):
        """ Load the model from the provided file
        """
        if not self.is_classical_regressor:
            self.dic_models = {}
            for quants in self.quantiles:
                q_low, q_high = quants[0], quants[1]

                model_quantile_low, model_quantile_high = clone(self.model), clone(self.model)

                model_quantile_low.set_params(quantile=q_low / 100)
                model_quantile_high.set_params(quantile=q_high / 100)

                model_quantile_low.fit(X, y)
                model_quantile_high.fit(X, y)

                self.dic_models[q_low] = model_quantile_low
                self.dic_models[q_high] = model_quantile_high

        else:
            self.model.fit(X, y)

        return self

    def predict(self, X, *args):
        """ Estimate the conditional low and high quantiles given the loaded prediction file, and the dates to predicti

        Parameters
        ----------
        X : numpy arrau, containing values from which to predict

        Returns
        -------
        ret_val : numpy array of estimated conditional quantiles (n, 2 * n_intervals)

        """
        if not self.is_classical_regressor:
            ret_val = np.zeros((X.shape[0], 2 * len(self.quantiles)))
            for i, quants in enumerate(self.quantiles):
                q_low, q_high = quants[0], quants[1]
                model_quantile_low, model_quantile_high = self.dic_models[q_low], self.dic_models[q_high]

                lower = model_quantile_low.predict(X)
                upper = model_quantile_high.predict(X)

                ret_val[:, i] = lower
                ret_val[:, -(i + 1)] = upper

        else:
            ret_val = self.model.predict(X)

        return ret_val


###############################################################################
# Custom External Regressor Adapater
###############################################################################
# TODO : Create a more robust estimator :
class CustomExternalRegressorAdapter(RegressorAdapter):
    """ Conditional quantile estimator that is already trained and predicting external value given to it
    Only reading csv for the moment
    References
    ----------
    """

    def __init__(self,
                 model,
                 fit_params=None,
                 quantiles=None,
                 params=None):
        """ Initialization

        Parameters
        ----------
        model : None, unused parameter (for compatibility with nc class)
        fit_params : None, unused parameter (for compatibility with nc class)
        quantiles : numpy array, low and high quantile levels in range (0,100)
        params : dictionary of parameters
                params["location_file"] : File with prediction, id and quantile table. The file msut be at csv format
                params["delimiter"] : Delimiter to read file from csv format

        """
        super(CustomExternalRegressorAdapter, self).__init__(model, fit_params)
        # Instantiate model
        self.quantiles = quantiles
        self.location_file = params['location_file']
        self.params = params
        self.dic = {}

    def fit(self, *args):
        """ Load the model from the provided file
        """
        X = read_csv(os.path.join(self.location_file, 'x.csv'))
        y = read_csv(os.path.join(self.location_file, 'y.csv'))
        assert X.shape[0] == y.shape[0]
        for i in range(X.shape[0]):
            x = np.floor(X.iloc[i].to_numpy().copy())
            if y.shape[1] == 1:
                self.dic[hash(x.tobytes())] = y.iloc[i][0]
                self.ydim = 1
            else:
                self.dic[hash(x.tobytes())] = y.to_numpy()[i]

        self.ydim = y.shape[1]

        return self

    def predict(self, X, *args):
        """ Estimate the conditional low and high quantiles given the loaded prediction file, and the dates to predicti

        Parameters
        ----------
        ids : numpy array of ids for which to give the predictions

        Returns
        -------
        ret_val : numpy array of estimated conditional quantiles (n, 2 * n_intervals)

        """
        if self.ydim == 1:
            ret_val = np.zeros(X.shape[0])
        else:
            ret_val = np.zeros((X.shape[0], self.ydim))

        for i in range(X.shape[0]):
            x = np.floor(X[i].copy())
            try:
                ret_val[i] = self.dic[hash(x.tobytes())]
            except:
                raise ValueError('Given array not in the prediction set')

        return ret_val
