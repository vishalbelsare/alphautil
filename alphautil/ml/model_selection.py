import numpy as np


def train_test_roll(array, tr_samples, te_samples, roll=None):
    """
    Rolling train-period

        array
        --------------
        tr  te
        ----++
          ----++
            ----++
              ......

    Returns
    -------
    splitting : list

    Examples
    --------
    >>> array = range(50)
    >>> for i_tr, i_te in train_test_roll(array, 20, 10):
    ...     print(i_tr, i_te)
    range(0, 20) range(20, 30)
    range(10, 30) range(30, 40)
    range(20, 40) range(40, 50)
    """
    if roll is None:
        roll = te_samples
    i = 0
    splitting = []
    while i + tr_samples + te_samples <= len(array):
        index_tr = array[i : i + tr_samples]
        index_te = array[i + tr_samples : i + tr_samples + te_samples]
        splitting.append((index_tr, index_te))
        i += roll
    return splitting
