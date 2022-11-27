import numpy as np
from numpy import average, zeros, mean, log
from scipy.stats import chi2, norm 


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
