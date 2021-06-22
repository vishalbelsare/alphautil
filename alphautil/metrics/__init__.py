import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_consistent_length
from sklearn.utils import check_X_y


def corr_score(x: np.array, y: np.array) -> float:
    """
    Return correlation coefficient between x and y.

    Parameters
    ----------
    x : 1d array_like
    y : 1d array_like

    Returns
    -------
    slope : float
    """
    check_consistent_length(x, y)

    return np.corrcoef(x, y)[0, 1]


def r2_score(x: np.array, y: np.array) -> float:
    """
    Return R^2 of linear regression from x to y.

    Parameters
    ----------
    x : 1d array_like
    y : 1d array_like

    Returns
    -------
    slope : float
    """
    X = x.reshape(-1, 1)
    check_X_y(X, y)

    return LinearRegression().fit(X, y).score(X, y)


def slope_score(x: np.array, y: np.array) -> float:
    """
    Return slope of linear regression from x to y.

    Parameters
    ----------
    x : 1d array_like
    y : 1d array_like

    Returns
    -------
    slope : float
    """
    check_consistent_length(x, y)

    return LinearRegression().fit(X, y).coef_[0]


def autocorr(x: np.array, max_lag=10):
    """
    Return auto-correlation function.

    Parameters
    ----------
    x : 1d array_like

    Returns
    -------
    autocorr : np.array, shape (max_lag,)
    """
    assert max_lag < x.shape[0]

    return np.array(
        [np.corrcoef(x[lag:], x[:-lag])[0, 1] for lag in range(1, max_lag + 1)]
    )
