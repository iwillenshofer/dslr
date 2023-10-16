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
    EPOCHS=3000
    LEARNING_RATE = 1

    def __init__(self, df: DataFrame, features: [str], label: str, graph=False):
        self.e = self.calculate_euler()
        self.graph = graph
        self.dataset = df
        self.features = features;
        self.label = label
        self.classes = df[label].unique()
        self.classes.sort()
        self.weight_matrix = [[0 for x in self.classes] for i in features]
        self.learning_rate = self.LEARNING_RATE
        self.epochs = self.EPOCHS
        #if graph:
        #    self.graph_sigmoid()
        weights = self.train()
        
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
        value = 1 / (1 + self.e ** -z)
        return (value)



    def cost_function(self, ys: [float], hs: [float]):
        """
        Cost Function: 
        J(θ) = -1/m * sum(y * log(h) + (1 - y) * log(1 - h))
        J(θ) is the cost function that needs to be minimized during the training of logistic regression.
        m is the number of training examples.
        h is the predicted probability that example Xi belongs to the positive class (sigmod function) on x * class.
        y is the actual label (0 or 1) for example Xi
        """
        y_costs = [ys[i] * math.log(hs[i]) + (1 - ys[i]) * math.log(1 - hs[i]) for i, _ in enumerate(hs)]
        cost = (-1 / len(ys)) * sum(y_costs)
        return cost



    def one_vs_all(self, clas: str):
        """
        creates a new column called 'class' (if it does not exist)
        sets all its values to 0
        sets its value to 1 if its label row value == class
        """
        self.dataset['class'] = 0
        self.dataset.loc[self.dataset[self.label] == clas, 'class'] = 1


    def train(self):
        """
        for each class, create a one_vs_all and get the weights of its features
        using a gradient descent
        """
        dic = {}
        df = self.dataset[[LABEL] + FEATURES].copy()
        for c in self.classes:
            dic[c] = {}
            self.one_vs_all(c)
            #ratio = 1 / (stats.max(data[feature]) - stats.min(data[feature]))
            #stats.normalize_dataframe(data)
            weights = self.gradient_descent() #* ratio
            dic[c] = weights
        for key in dic:
            print(dic[key])
        return (dic)
    

    def gradient_descent(self):
        import numpy as np
        m = len(self.dataset)
        thetas = np.random.rand(m)
        for i in range(self.epochs):
            z = [sum([thetas[j] * self.dataset[feature][i] for j, feature in enumerate(self.features)]) for i in range(m)]
            h = [self.sigmoid(zz) for zz in z]
            gradients = [sum([(h[i] -  self.dataset['class'][i]) * self.dataset[feature][i] for i in range(m)]) / m for _, feature in enumerate(self.features)]
            thetas = [thetas[j] - self.learning_rate * gradients[j] for j, _ in enumerate(self.features)]
            cost = self.cost_function(self.dataset['class'], h)
            print(cost, thetas[0])
        return thetas





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
        for feature in FEATURES:
            ds[feature].fillna(stats.mean(ds[feature]), inplace=True)
        stats.normalize_dataframe(ds)
#        print(stats.Describe(ds))
        LogisticRegression(ds, FEATURES, LABEL, graph=True)
    #except Exception as error:
    #    print(f"{type(error).__name__}: {error}")


if __name__ == "__main__":


    main()
    exit()
    # Import necessary libraries
    import numpy as np
    from sklearn.model_selection import train_test_split
    import sklearn.linear_model as lm
    from sklearn.metrics import accuracy_score, confusion_matrix

    ds = read_csv("datasets/dataset_train.csv", index_col=INDEX_COL)
    ds.fillna(0, inplace=True)
    stats.normalize_dataframe(ds)
    # Generate some random data for demonstration purposes
    np.random.seed(42)
    X = ds[FEATURES]
    y = ds[[LABEL]]

    model = lm.LogisticRegression(multi_class='ovr', solver='lbfgs')
    model.fit(X, y)

    # Make predictions on the test set
    #y_pred = model.predict(X_test)

    # Display intercept and weights for each class
    print("Intercepts:", model.intercept_)
    print("Weights for each class:")
    print(model.classes_)
    print(model.feature_names_in_)
    print(model.coef_)
    print(model)
    # Evaluate the model
    #accuracy = accuracy_score(y_test, y_pred)
    #conf_matrix = confusion_matrix(y_test, y_pred)

    # Print results
    #print(f"Accuracy: {accuracy:.2f}")
    #print("Confusion Matrix:")
    #print(conf_matrix)


