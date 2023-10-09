def calculate_percentile(data: list, nth: int) -> any:
    """
    assumes it is receiving sorted data
    receives a data series and the desired nth percentile.
    """
    n = len(data)
    if not n:
        return float('nan')
    if nth > 100 or nth < 0:
        raise ValueError("Percentiles must be in the range [0, 100]")
    lower_index = (nth / 100) * (n - 1)
    rest = lower_index % 1
    lower_index = int(lower_index // 1)
    if (rest.is_integer() and rest == 0):
        return (data[lower_index])
    upper_index = lower_index + 1
    return (data[lower_index]
            + ((data[upper_index] - data[lower_index]) * rest))


def percentile(data: list, nth: any) -> any:
    """
    receives a data serie and a percentile or a list of percentiles
    """
    a = []
    d = sorted(data)
    if (type(nth) is int):
        return (calculate_percentile(d, nth))
    for i in nth:
        a.append(calculate_percentile(d, i))
    return (a)
