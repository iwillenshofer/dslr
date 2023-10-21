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
    """verifies if 2 argument is passed"""
    argc = len(sys.argv)
    if argc < 3:
        raise AssertionError(
            "Test file and weights file are expected as arguments")
    elif argc > 3:
        raise AssertionError("Too many arguments")


def main():
    """
    loads input file, process it and applies model to it.
    """
    try:
        validate_args()
        ds = read_csv(sys.argv[1], index_col=INDEX_COL)
        for feature in FEATURES:
            ds[feature].fillna(stats.mean(ds[feature]), inplace=True)
        logreg = LogisticRegression(verbose=True, fit_intercept=True)
        logreg.load_model(sys.argv[2])
        logreg.predict(
            ds[FEATURES], save=True, filename='houses.csv', labelname=LABEL)
    except Exception as error:
        print(Exception.__name__ + ":", error)


if __name__ == "__main__":
    main()
