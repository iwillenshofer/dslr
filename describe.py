import sys
from tools import stats
from pandas import read_csv

# Defines the index column.
# may be set as the column name, 0 for no index, or None, for auto number
INDEX_COL = 'Index'


def validate_args():
    """verifies if 1 argument is passed"""
    argc = len(sys.argv)
    if argc < 2:
        raise AssertionError("File name is expected as first argument")
    elif argc > 2:
        raise AssertionError("Too many arguments")


def main():
    """loads file and process it, displaying its data description"""
    try:
        validate_args()
        ds = read_csv(sys.argv[1], index_col=INDEX_COL)
        print(stats.Describe(ds))
    except Exception as error:
        print(f"{type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
