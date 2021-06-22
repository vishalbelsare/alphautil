def winsorize(*arrays, by=None, values=None, quantiles=None):
    """
    Remove outliers in arrays.

    Parameters
    ----------
    *arrays : iterable[array_like]
        Arrays to winsorize.
    by : array_like, Optional
        Winsorize elements at the positions where
        the value of `by` is outlier.
        If not given, the first array.
    values : tuple[float | None], Optional
        (min, max)
        If None, do not winsorize
    quantiles : tuple[float | None], Optional
        (lower, upper)
        If None, do not winsorize

    Returns
    -------
    winsorized_arrays : list[array_like]
    """
    if by is None:
        by = arrays[0]

    if values is not None:
        min_v, max_v = values
        min_v = -float("inf") if min_v is None else min_v
        max_v = +float("inf") if max_v is None else max_v
    elif quantiles is not None:
        min_q, max_q = quantiles
        min_q = 0 if min_q is None else min_q
        max_q = 1 if max_q is None else max_q
        min_v = np.quantile(by, min_q)
        max_v = np.quantile(by, max_q)
    else:
        min_v = min(by)
        max_v = max(by)
    return [a[(min_v < by) & (by < max_v)] for a in arrays]
