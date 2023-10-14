#
# Awesome source of information
# https://medium.com/analytics-vidhya/logistic-regression-from-scratch-multi-classification-with-onevsall-d5c2acf0c37c
#

import sys
import math
from tools import stats
from pandas import read_csv, DataFrame
from matplotlib import pyplot as plt

# Defines the index column.
# may be set as the column name, 0 for no index, or None, for auto number
INDEX_COL = 'Index'

FEATURES = [
    'Arithmancy',
    'Astronomy',
    'Herbology',
    'Defense Against the Dark Arts',
    'Divination',
    'Muggle Studies',
    'Ancient Runes',
    'History of Magic',
    'Transfiguration',
    'Potions',
    'Care of Magical Creatures',
    'Charms',
    'Flying'
]

LABEL = 'Hogwarts House'

class LogisticRegression:
    EPOCHS=1000
    LEARNING_RATE = 1

    def __init__(self, df: DataFrame, features: [str], label: str, graph=False):
        self.e = self.calculate_euler()
        self.graph = graph
        self.dataset = df
        self.features = features;
        self.label = label
        self.classes = df[label].unique()
        self.weight_matrix = [[0 for x in self.classes] for i in features]
        #if graph:
        #    self.graph_sigmoid()
        self.train()
        
    def graph_sigmoid(self):
        rng = [x / 10 for x in range(-50, 50)]
        sigmod_values = [self.sigmoid(i) for i in rng]
        derivative_values = [s * (1 - s) for s in sigmod_values]
        plt.plot(sigmod_values)
        plt.plot(derivative_values)
        plt.plot([0.5 for i in sigmod_values])
        plt.show()


    def calculate_euler(self):
        e = 1.0
        factorial = 1
        for i in range(1, 100):
            factorial *= i
            e += 1.0 / factorial
        return e            


    def sigmoid(self, z: float):
        return (1 / (1 + self.e ** -z))


    def cost_function(self, data: DataFrame, feature: str):
        X = list(data[feature])
        y = list(data['class'])

        a = sum([y[i] * math.log(self.sigmoid( X[i] * y[i] )) + (1 - y[i]) * math.log(1 - self.sigmoid(X[i] * y[i]))  for i, _ in enumerate(X)])
        return (-1/len(X) * a)


    def one_vs_all(self, feature: str, clas: str):
        """
        returns a dataframe containing only the feature and label cols,
        setting the selected class as 1 and all the others as 0.
        nans are dropped.
        """
        df = self.dataset[[feature, self.label]].copy()
        df['class'] = 0
        df.loc[df[self.label] == clas] = 1
        df.drop(self.label, axis=1, inplace=True)
        df.dropna(inplace=True)
        return (df)

#    def gradient_descent(self, data, feature, clas):


    def train(self):
        for feature in self.features:
            for c in self.classes:
                data = self.one_vs_all(feature, c)
                print(data)
                print(self.cost_function(data, feature))




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
        ds = read_csv(sys.argv[1], index_col=INDEX_COL)
        print(stats.Describe(ds))
        LogisticRegression(ds, FEATURES, LABEL, graph=True)
    #except Exception as error:
    #    print(f"{type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
