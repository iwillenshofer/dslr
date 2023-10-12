def mean(d: list) -> float:
    """
    calculates the mean value
    """
    data = [x for x in d if x == x]
    size = len(data)
    if not size:
        return float('nan')
    return (sum(data) / float(size))


def stdev(d: list, ddof=1) -> float:
    """
    calculates standard deviation
    ddof = 1 for sample
    set ddof = 0 for populaton
    """
    data = [x for x in d if x == x]
    size = len(data)
    if size < 2:
        return (float('nan'))
    m = mean(data)
    variance = sum([(float(i) - m) ** 2 for i in data]) / (float(size) - ddof)
    stddev = variance ** .5
    return (stddev)
