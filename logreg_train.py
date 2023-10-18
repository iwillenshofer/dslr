#
# Awesome source of information
# https://medium.com/analytics-vidhya/logistic-regression-from-scratch-multi-classification-with-onevsall-d5c2acf0c37c
#

import sys
import json
import math
import numpy as np
from tools import stats
from pandas import read_csv, DataFrame, to_datetime
from matplotlib import pyplot as plt

from LogisticRegression import LogisticRegression

# Defines the index column.
# may be set as the column name, 0 for no index, or None, for auto number
INDEX_COL = 'Index'

FEATURES = [
    'Best Hand',
    'Birthday',
#    'Arithmancy',
    'Astronomy',
    'Herbology',
    'Defense Against the Dark Arts',
    'Divination',
    'Muggle Studies',
    'Ancient Runes',
    'History of Magic',
    'Transfiguration',
    'Potions',
#    'Care of Magical Creatures',
    'Charms',
    'Flying'
]

LABEL = 'Hogwarts House'









def validate_args():
    """verifies if 1 argument is passed"""
    argc = len(sys.argv)
    if argc < 2:
        raise AssertionError("File name is expected as first argument")
    elif argc > 2:
        raise AssertionError("Too many arguments")


def main():
        """loads file and process it, displaying its data description"""
    #try:
        validate_args()
        
        
        ###LOADING THE INPUT FILE, TREATING IT, GENERATING A MODEL AND SAVING IT...
        ds = read_csv(sys.argv[1], index_col=INDEX_COL)
        ds.replace('Right', 0, inplace=True)
        ds.replace('Left', 1, inplace=True)
        ds["Birthday"] = to_datetime(ds["Birthday"]).dt.strftime("%Y%m%d").astype(int)
        for feature in FEATURES:
            ds[feature].fillna(stats.mean(ds[feature]), inplace=True)
        logreg = LogisticRegression(verbose=True)
        logreg.fit(ds[FEATURES], ds[LABEL])
        logreg.save_model()
        
        
        ###LOADING THE MODEL AND RUNING AGAINST THE ORIGINAL DATASET (stupid, I know, but just to check if everything is going smoothly)
        logreg2 = LogisticRegression(verbose=True, fit_intercept=True)
        logreg2.load_model()
        res = logreg2.predict(ds[FEATURES])
        cost = logreg2.score(ds[FEATURES], ds[LABEL])
        print(res)
        print(cost)
 


		### NOW, OPEN THE TEST DATASET, TREAT IT, AND PREDICT IT
        ds = read_csv("datasets/dataset_test.csv", index_col=INDEX_COL)
        ds.replace('Right', 0, inplace=True)
        ds.replace('Left', 1, inplace=True)
        ds["Birthday"] = to_datetime(ds["Birthday"]).dt.strftime("%Y%m%d").astype(int)
        for feature in FEATURES:
            ds[feature].fillna(stats.mean(ds[feature]), inplace=True)
        logreg2 = LogisticRegression(verbose=True, fit_intercept=True)
        logreg2.load_model()
        res = logreg2.predict(ds[FEATURES])
        
        
        
        ### NOW COMPARE THE PREDICTION WITH TRUTH.CSV
        truth = read_csv("datasets/dataset_truth.csv", index_col=INDEX_COL)
        cost = logreg2.score(ds[FEATURES], truth[LABEL])
        print(res)
        print(cost)        


if __name__ == "__main__":
    main()


