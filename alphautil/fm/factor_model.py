import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class FactorModel:
    """Factor model.

    Args:
        factors (pandas.DataFrame): The index stands for the timestamps.
            Each column is a time-series of factor returns.

    Attributes:
        beta_ (DataFrame): ...
            shape (S, F) where S is the number of stocks and F is the number of factors.

    Examples:

        >>> # TODO
    """

    def __init__(self, factors, risk_free_rate=0.0, factor_names=None, freq="D"):
        assert risk_free_rate == 0, "not supported"
        assert freq == "D", "not supported"

        # make factors daily and ffill
        self._begin = factors.index[0]
        self._end = factors.index[-1]
        factors = factors.reindex(pd.date_range(self._begin, self._end)).ffill()
        self.factors = factors

    def __repr__(self):
        period = f"{self._begin.date()} TO {self._end.date()}"
        return f"{self.__class__.__name__}({self.factor_names}, {period})"

    @property
    def index(self):
        return self.factors.index

    @property
    def factor_names(self):
        return list(self.factors.columns)

    @property
    def n_factors(self):
        return len(self.factor_names)

    def fit(self, X, y=None):
        """
        - X : DataFrame
            The index stands for the timestamps.
            Each column is a time-series of factor returns.
        """
        f = self.factors.loc[X.index]

        self.lr = LinearRegression(fit_intercept=False).fit(X=f, y=X)
        self.beta_ = pd.DataFrame(
            self.lr.coef_, index=X.columns, columns=self.factor_names
        )

        return self

    def transform(self, X):
        """

        Args:
            X (pandas.DataFrame): Stock returns

        Returns:
            alpha (pandas.DataFrame): resudual
        """
        f = self.factors.loc[X.index]
        return X - f.dot(self.beta_.T)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def _align_timestamps(self, df1, df2):
        index = df1.index.intersection(df2.index)
        return df1.loc[index], df2.loc[index]
