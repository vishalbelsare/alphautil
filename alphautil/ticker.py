import os

import pandas


DEFAULT_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities"
    "/misc/tvdivq0000001vg2-att/data_j.xls"
)


def _fetch_data_j(
    path="data_j.csv",
    url=DEFAULT_URL
) -> None:
    pandas.read_excel(url).to_csv(path)


def get_t2n(path="data_j.csv", prefix="", suffix="") -> dict:
    """
    Get dict object from ticker to name.

    Args:
        path (str): ...
        prefix (str, optional): ...
        suffix (str, optional): ...

    Returns:
        dict
    """
    if not os.path.exists(path):
        _fetch_data_j(path)

    dataframe = pandas.read_csv(path)
    tickers = dataframe["コード"].astype(str)
    tickers = [prefix + _ for _ in tickers]
    tickers = [_ + suffix for _ in tickers]
    names = dataframe["銘柄名"]
    names = [_.replace("\u3000", " ") for _ in names]
    return dict(zip(tickers, names))
