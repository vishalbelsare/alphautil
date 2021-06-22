import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def axline(x=None, y=None, **kwargs):
    if x is not None:
        plt.axvline(x, **kwargs)
    if y is not None:
        plt.axhline(y, **kwargs)


def score_linreg(x, y, scores={...}):
    return dict(...)


def plot_linreg(x: np.ndarray, y: np.ndarray, label="auto", **kwargs) -> None:
    """Line-plot linear regression line.

    Args:
        x (np.ndarray): Predictor variable.
        y (np.ndarray): Predicted variable.
        label (str, default="auto")
            If "auto", set automatically.
        **kwargs
    """
    lr = LinearRegression().fit(x.reshape(-1, 1), y)
    xlim = np.linspace(min(x), max(x))
    ylim = lr.predict(xlim.reshape(-1, 1))
    if label == "auto":
        slope = lr.coef_[0]
        r2 = lr.score(x.reshape(-1, 1), y)
        corr = np.corrcoef(x, y)[0, 1]
        label = f"Linear Regression (slope={slope:.2e}, R^2={r2:.2e}, corr={corr:.2e})"
    plt.plot(xlim, ylim, label=label, **kwargs)


def plot_quantile_means(
    x: np.ndarray, y: np.ndarray, bins=10, label="auto", **kwargs
) -> None:
    """Plot mean of each quantile groups.

    Args:
        x (np.ndarray): Predictor variable.
        y (np.ndarray): Predicted variable.
        bins (int, default=10): Number of bins.
        label (str, default "auto"): If "auto", set automatically.
        **kwargs: Passed to `matplotlib.pyplot.plot`.
    """
    qs = np.linspace(0.0, 1.0, bins + 1)
    quantiles = [np.quantile(x, q) for q in qs]
    for i, (begin, end) in enumerate(zip(quantiles, quantiles[1:])):
        mean = y[(begin <= x) & (x < end)].mean()
        xlim = np.linspace(begin, end)
        ylim = np.broadcast_to(mean, xlim.shape)
        if label == "auto":
            label_ = f"Quantile {i + 1}: mean={mean:.4e}"
        else:
            label_ = label
        plt.plot(xlim, ylim, label=label_, **kwargs)


import matplotlib
import seaborn


class AppleHIGColor:
    BLUE = "#007aff"
    BROWN = "#a2845e"
    GRAY = "#8e8e93"
    GREEN = "#28cd41"
    INDIGO = "#5856d6"
    ORANGE = "#ff9500"
    PINK = "#ff2d55"
    PURPLE = "#af52de"
    RED = "#ff3b30"
    TEAL = "#55bef0"
    YELLOW = "#ffcc00"

    @classmethod
    def cycle(cls):
        return matplotlib.cycler(
            color=[
                cls.BLUE,
                cls.ORANGE,
                cls.GREEN,
                cls.RED,
                cls.PURPLE,
                cls.BROWN,
                cls.INDIGO,
                cls.TEAL,
                cls.YELLOW,
            ]
        )


def set_color(style="apple-hig") -> None:
    assert style == "apple-hig"
    style = {}
    if style == "apple-hig":
        style["axes.prop_cycle"] = AppleHIGColor.cycle()
    matplotlib.rcParams.update(style)


def set_seaborn_style(style=None):
    seaborn.set_style(style)


def set_dpi(dpi=300) -> None:
    matplotlib.rcParams.update({"figure.dpi": dpi})


def set_savefig(bbox="tight", pad_inches=0.1):
    matplotlib.rcParams.update({"savefig.bbox": bbox, "savefig.pad_inches": pad_inches})
