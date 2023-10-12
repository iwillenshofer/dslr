from pandas import DataFrame
from pandas.api.types import is_numeric_dtype
from .stdev import mean, stdev


def min(lst: [float]):
    """returns the minimum value of a list"""
    value = None
    for i in lst:
        if i == i and (value is None or i < value):
            value = i
    if value is None:
        raise AssertionError("No numeric values in list")
    return (value)


def max(lst: [float]):
    """returns the maximum value of a list"""
    value = None
    for i in lst:
        if i == i and (value is None or i > value):
            value = i
    if value is None:
        raise AssertionError("No numeric values in list")
    return (value)


def normalize_stdev(lst: [float]):
    """"""
    mean_value = mean(lst)
    stdev_value = stdev(lst)
    norm = [((x - mean_value) / stdev_value) if x == x else x for x in lst]
    return (norm)


def normalize_minmax(lst: [float]):
    """normalize data using minmax"""
    x_min = min(lst)
    x_max = max(lst)
    norm = [((x - x_min) / (x_max - x_min)) if x == x else x for x in lst]
    return (norm)


def normalize_dataframe(ds: DataFrame, method='minmax'):
    for i in ds:
        if is_numeric_dtype(ds[i]):
            if method == 'minmax':
                ds[i] = normalize_minmax(ds[i])
            else:
                ds[i] = normalize_stdev(ds[i])
