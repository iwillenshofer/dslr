import sys
from tools import stats
from pandas import read_csv
from tools.LogisticRegression import LogisticRegression

INDEX_COL = 'Index'

# Features not considered Removed:
# 'Birthday'
# 'Best Hand'
# 'Arithmancy'
# 'Care of Magical Creatures'

FEATURES = [
    'Astronomy',
    'Herbology',
    'Defense Against the Dark Arts',
    'Divination',
    'Muggle Studies',
    'Ancient Runes',
    'History of Magic',
    'Transfiguration',
    'Potions',
    'Charms',
    'Flying'
]

LABEL = 'Hogwarts House'


def validate_args():
    """
    verifies if 1 argument is passed
    """
    argc = len(sys.argv)
    if argc < 2:
        raise AssertionError("File name is expected as first argument")
    elif argc > 2:
        raise AssertionError("Too many arguments")


def main():
    """
    loads input file, process it and generates a model.
    """
    try:
        validate_args()
        ds = read_csv(sys.argv[1], index_col=INDEX_COL)
        for feature in FEATURES:
            ds[feature].fillna(stats.mean(ds[feature]), inplace=True)
        logreg = LogisticRegression(verbose=True)
        logreg.fit(ds[FEATURES], ds[LABEL], save_history=False, epochs=300)
        logreg.graph()
        logreg.save_model()
    except Exception as error:
        print(Exception.__name__, error)


if __name__ == "__main__":
    main()
