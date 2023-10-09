from .stdev import mean


def fill_na(lst: list, value='mean'):
    """
    only mean method has been implemented.
    to fill with another value (0, -9999), just set value to it
    """
    number_values = [i for i in lst if i == i]
    if (value == 'mean'):
        replace = mean(number_values)
    else:
        replace = value
    replaced_values = [(i if i == i else replace) for i in lst]
    return (replaced_values)


def remove_na(lst: list):
    """
    removes nan from a list
    """
    return ([i for i in lst if i == i])
