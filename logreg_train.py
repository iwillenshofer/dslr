#
# Awesome source of information
# https://medium.com/analytics-vidhya/logistic-regression-from-scratch-multi-classification-with-onevsall-d5c2acf0c37c
#

import sys
import math
import numpy as np
from tools import stats
from pandas import read_csv, DataFrame, to_datetime
from matplotlib import pyplot as plt

class LogisticRegression:
    DEFAULT_MAX_EPOCHS=70000
    DEFAULT_LEARNING_RATE = .15
    DEFAULT_DELTA_THRESHOLD = 1e-7

    def __init__(self, verbose=False, fit_intercept=True):
        self.y = None
        self.X = None
        self.weights = None
        self.biases = None
        self.classes = None
        self.epochs = self.DEFAULT_MAX_EPOCHS
        self.learning_rate = self.DEFAULT_LEARNING_RATE
        self.min_values = None
        self.max_values = None
        self.fit_intercept = True
        self.verbose = verbose
        self.fit_intercept = fit_intercept
        self.delta_threshold = self.DEFAULT_DELTA_THRESHOLD

    def __log(self, *values: object):
        if self.verbose:
            print(*values)

    def __sigmoid(self, z: float):
        value = 1 / (1 + np.exp(-z))
        return (value)

    def __cost_function(self, ys: [float], hs: [float]):
        """
        Cost Function: Cross Entropy
        J(θ) = -1/m * sum(y * log(h) + (1 - y) * log(1 - h))
        J(θ) is the cost function that needs to be minimized during the training of logistic regression.
        m is the number of training examples.
        h is the predicted probability that example Xi belongs to the positive class (sigmod function) on x * class.
        y is the actual label (0 or 1) for example Xi
        """
        y_costs = [ys[i] * math.log(hs[i]) + (1 - ys[i]) * math.log(1 - hs[i]) for i, _ in enumerate(hs)]
        cost = (-1 / len(ys)) * sum(y_costs)
        return cost


    def __one_vs_all(self, y: [str], clas: str):
        """
        creates a numpy array based on y values
        sets all its values to 0
        sets its value to 1 if its label value == clas
        """
        y_class = np.copy(y)
        y_class[y_class == clas] = 1
        y_class[y_class != clas] = 0
        y_class = y_class.astype(np.int64)
        return (y_class)


    def __normalize(self, X, min_values=None, max_values=None):
        self.min_vals = np.min(X, axis=0) if not min_values else min_values
        self.max_vals = np.max(X, axis=0) if not max_values else max_values
        X_scaled = (X - self.min_vals) / (self.max_vals - self.min_vals)
        return X_scaled


    def __gradient_descent(self, X, y):
        prev_cost = 0
        m = X.shape[0]
        thetas = np.random.rand(X.shape[1])
        np.random.seed()
        bias = np.random.rand(1) if self.fit_intercept else 0
        for iteration in range(self.epochs):
            z = np.dot(X, thetas) 
            z = z + bias
            h = self.__sigmoid(z)
            gradients = (1 / m) * np.dot((h - y), X)
            thetas -= self.learning_rate * gradients
            if self.fit_intercept:
                gradients_bias = (1 / m) * np.sum((h - y))
                bias -= self.learning_rate * gradients_bias
            cost = self.__cost_function(y, h)	
            if iteration > 0 and abs(cost - prev_cost) < self.delta_threshold:
                self.__log(f"Breaking @ iteration {iteration}.")
                break
            prev_cost = cost
        return(thetas, bias)


    def __train(self):
        """
        for each class, create a one_vs_all and get the weights of its features
        using a gradient descent
        """
        self.weights = {}
        self.biases = {}
        for c in self.classes:
            self.__log("Training for class:", c)
            self.weights[c] = {}
            y_c = self.__one_vs_all(self.y, c)
            weights, bias = self.__gradient_descent(self.X, y_c) #* ratio
            self.weights[c] = weights
            self.biases[c] = bias
        self.__log("Weights:", self.weights)
        self.__log("Biases:", self.biases)


    def fit(self, X, y, normalize=True, delta_threshold=DEFAULT_DELTA_THRESHOLD, max_iterations=DEFAULT_MAX_EPOCHS, learning_rate=DEFAULT_LEARNING_RATE):
        self.epochs = max_iterations
        self.learning_rate = learning_rate
        self.delta_threshold = delta_threshold
        self.X = X
        self.y = y
        self.classes = np.unique(self.y)
        self.classes.sort()
        if normalize:
            self.X = self.__normalize(self.X)
        self.__train()


    def save_model(self):
        pass
    
    def load_model(self):
        pass


    def accuracy(self):
        pass
    

    def predict(self, X, normalize=True):
        ds = DataFrame
        if normalize == True:
            X = self.__normalize(X, self.min_vals, self.max_vals)
        for c in self.weights:
            linear_pred = np.dot(X, self.weights[c])
            ds[c] = self.__sigmoid(linear_pred + self.biases[c])
        ds['prediction'] = ds[self.weights.keys()].idxmax(axis=1)
        ds = ds[['prediction']]
        self.__log(ds)
        
    
    
        #y = ds[LABEL].values
        #times_right = (ds[LABEL] == ds['prediction']).sum()
        #print(times_right / len(ds))



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
        ds = read_csv(sys.argv[1], index_col=INDEX_COL)
        ds.replace('Right', 0, inplace=True)
        ds.replace('Left', 1, inplace=True)
        ds["Birthday"] = to_datetime(ds["Birthday"]).dt.strftime("%Y%m%d").astype(int)
        for feature in FEATURES:
            ds[feature].fillna(stats.mean(ds[feature]), inplace=True)
        logreg = LogisticRegression(verbose=True, fit_intercept=True)
        logreg.fit(ds[FEATURES], ds[LABEL])
        
        
        
        
        
        return
        stats.normalize_dataframe(ds)
        
        
        logreg = LogisticRegression(ds, FEATURES, LABEL, graph=False)
        
        
        
        
        
        dp = read_csv("datasets/dataset_predict.csv", index_col=INDEX_COL)
        dp.replace('Right', 0, inplace=True)
        dp.replace('Left', 1, inplace=True)
        dp["Birthday"] = to_datetime(dp["Birthday"]).dt.strftime("%Y%m%d").astype(int)
        for feature in FEATURES:
            dp[feature].fillna(stats.mean(dp[feature]), inplace=True)
        stats.normalize_dataframe(dp)   
        logreg.predict(dp)
        
        
#        print(stats.Describe(ds))
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

    model = lm.LogisticRegression(multi_class='ovr', solver='liblinear', fit_intercept=False, verbose=1000)
    model.fit(X, y)

    # Make predictions on the test set
    #y_pred = model.predict(X_test)

    # Display intercept and weights for each class
    print("Intercepts:", model.intercept_)
    print("Weights for each class:")
    print("Iterations:", model.n_iter_)
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


