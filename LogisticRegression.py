import json
import math
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame


class LogisticRegression:
    DEFAULT_MAX_EPOCHS = 500
    DEFAULT_LEARNING_RATE = .15
    DEFAULT_DELTA_THRESHOLD = 1e-7

    def __init__(self, verbose=False, fit_intercept=True):
        """
        initializes a logistic regression model

        Args:
            verbose (bool, optional): Displays log. Defaults to False.
            fit_intercept (bool, opt.): calculates biases. Defaults to True.
        """
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
        self.history = None
        self.save_history = False

    def __log(self, *values: object):
        """
        prints if self.verbose is true.
        """
        if self.verbose:
            print(*values)

    def __sigmoid(self, z):
        """
        Computes the sigmoid function for a given input
        """
        value = 1 / (1 + np.exp(-z))
        return (value)

    def __cost_function(self, ys: [float], hs: [float]):
        """
        Cost Function: Cross Entropy
        J(θ) = -1/m * sum(y * log(h) + (1 - y) * log(1 - h))
        J(θ) is the cost function that needs to be minimized during the
         training of logistic regression.
        m is the number of training examples.
        h is the predicted probability that example Xi belongs to the
         positive class (sigmod function) on x * class.
        y is the actual label (0 or 1) for example Xi
        """
        epsilon = 1e-15 
        y_costs = [ys[i] * math.log(hs[i] + epsilon) + (1 - ys[i]) * math.log(1 - hs[i] + epsilon) for i, _ in enumerate(hs)]           
        cost = (-1 / len(ys)) * sum(y_costs)
        return cost

    def __one_vs_all(self, y: [str], clas: str):
        """
        creates a numpy array based on y values
        sets its value to 1 if its label value == clas else, to zero
        """
        y_class = np.where(y == clas, 1, 0)
        y_class = y_class.astype(np.int64)
        return (y_class)

    def __normalize(self, X, min_vals=None, max_vals=None):
        """
        Normalizes input data based on provided min and max values.
        if no min_max values are provided, calculates them from the provided X
        """
        self.min_values = np.min(X, axis=0) if min_vals is None else min_vals
        self.max_values = np.max(X, axis=0) if max_vals is None else max_vals
        X_scaled = (X - self.min_values) / (self.max_values - self.min_values)
        return X_scaled

    def __gradient_descent(self, X, y):
        """
        Performs gradient descent to train the model for a specific class.
        """
        history = []
        prev_cost = 0
        m = X.shape[0]
        thetas = np.zeros(X.shape[1])
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
            if self.save_history:
                history.append({'z': np.sort(z).tolist(),
                                'h': np.sort(h).tolist(),
                                'bias': bias.tolist(),
                                'weights': thetas.tolist(),
                                'cost': cost,
                                'gradients': gradients.tolist()})
        return thetas, bias, history

    def __train(self):
        """
        for each class, create a one_vs_all and get the weights of its
        features using a gradient descent
        """
        self.weights = {}
        self.biases = {}
        self.history = {}
        for c in self.classes:
            self.__log("Training for class:", c)
            self.weights[c] = {}
            y_c = self.__one_vs_all(self.y, c)
            weights, bias, history = self.__gradient_descent(self.X, y_c)
            self.weights[c] = weights
            self.biases[c] = bias
            self.history[c] = history
        self.__log("Weights:", self.weights)
        self.__log("Biases:", self.biases)

    def fit(self, X, y, normalize=True,
            delta_threshold=DEFAULT_DELTA_THRESHOLD,
            max_iterations=DEFAULT_MAX_EPOCHS,
            learning_rate=DEFAULT_LEARNING_RATE, save_history=False):
        """_summary_

        Args:
            X (numpy.ndarray): Data to be trained (features x samples)
            y (numpy.ndarray): True label values
            normalize (bool, optional): Defaults to True.
            delta_threshold (_type_, optional): Defaults to 1e-7.
            max_iterations (_type_, optional): Defaults to 500.
            learning_rate (_type_, optional): Defaults to 0.15.
            save_history (bool, optional):
             wether to save the history along with the weights.
             Defaults to True.
        """
        self.epochs = max_iterations
        self.learning_rate = learning_rate
        self.delta_threshold = delta_threshold
        self.X = X
        self.y = y
        self.classes = np.unique(self.y)
        self.classes.sort()
        if normalize:
            self.X = self.__normalize(self.X)
        if save_history:
            self.save_history = True
        self.__train()

    def save_model(self, filename='model.json'):
        """
        Saves the trained model's weights, biases, classes,
        and other information to a JSON file.

        Args:
            filename (str, optional): name of the exported file.
             Defaults to 'model.json'.
        """
        dic = {}
        try:
            dic['weights'] = None if self.weights is None else {
                k: self.weights[k].tolist() for k in self.weights.keys()}
            dic['biases'] = None if self.biases is None else {
                k: self.biases[k].tolist() for k in self.biases.keys()}
            if self.save_history:
                dic['history'] = self.history
            dic['min_values'] = None if self.min_values is None \
                else self.min_values.tolist()
            dic['max_values'] = None if self.max_values is None \
                else self.max_values.tolist()
            with open(filename, "w") as outfile:
                outfile.write(json.dumps(dic, indent=4))
        except Exception:
            raise AssertionError(f"Could not save file {filename}")

    def load_model(self, filename='model.json'):
        """
        Loads the trained models's information

        Args:
            filename (str, optional): filename of the model to be loaded.
             Defaults to 'model.json'.
        """
        dic = {}
        try:
            with open(filename, 'r') as infile:
                dic = json.load(infile)
            self.weights = None if 'weights' not in dic else {
                k: np.array(dic['weights'][k]) for k in dic['weights'].keys()}
            self.biases = None if 'biases' not in dic else {
                k: np.array(dic['biases'][k]) for k in dic['biases'].keys()}
            self.classes = None if 'weights' not in dic else np.array(
                [k for k in dic['weights'].keys()])
            self.min_values = None if 'min_values' not in dic else np.array(
                dic['min_values'])
            self.max_values = None if 'max_values' not in dic else np.array(
                dic['max_values'])
            self.history = None if 'history' not in dic else dic['history']
        except Exception:
            raise AssertionError(f"Could not load model from '{filename}'")

    def score(self, X, y, normalize=True):
        """
        using the current model, predicts results for X and compares with y

        Args:
            X (numpy.ndarray): Data to be trained (features x samples)
            y (numpy.ndarray): True label values
        """
        ds = self.predict(X, labelname='prediction', normalize=normalize)
        ds['true'] = y
        times_right = (ds['prediction'] == ds['true']).sum()
        result = times_right / len(ds)
        return (result)

    def predict(self, X, normalize=True, save=True,
                filename='predictions.csv', labelname='Predictions'):
        """
        predicts the data based on a previously trained model

        Args:
            X (np.ndarray): values to be predicted (features x instances)
            normalize (bool, optional): wether X should be normalized on
             the same scale as the fitted model. Defaults to True.
            save (bool, optional): saves the prediction. Defaults to True.
            filename (str, optional): filename to be saved into.
             Defaults to 'predictions.csv'.
            labelname (str, optional): label of the predicted row.
             Defaults to 'Predictions'.
        """
        ds = DataFrame()
        if not self.weights:
            raise AssertionError("Model has not been trained yet")
        try:
            if normalize:
                X = self.__normalize(X, self.min_values, self.max_values)
            for c in self.weights:
                linear_pred = np.dot(X, self.weights[c]) + self.biases[c]
                ds[c] = self.__sigmoid(linear_pred)
            ds[labelname] = ds[self.weights.keys()].idxmax(axis=1)
            ds = ds[[labelname]]
            self.__log(ds)
        except Exception as error:
            raise AssertionError(f"Could not predict values: {error}")
        if save:
            try:
                ds.to_csv(filename, index_label='Index')
            except Exception:
                raise AssertionError("Could not save prediction into file")
        return ds

    def graph(self):
        """
        uses history to graph.
        history must be saved when fitting the data.
        """
        if not self.history:
            raise AssertionError(
                "Nothing to Display. Use save_history during training")
        colorsmap = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        colors = {k: colorsmap[i] for i, k in enumerate(self.history.keys())}
        fig, axs = plt.subplots(4)
        costs = {k: [c['cost'] for c in self.history[k]] for k in self.history}
        axs[0].set_xlabel("weighted probability")
        axs[0].set_ylabel("sigmoid function")
        axs[1].set_xlabel("iteractions")
        axs[1].set_ylabel("cost function")
        axs[2].set_xlabel("features")
        axs[2].set_ylabel("gradient")
        axs[3].set_xlabel("features")
        axs[3].set_ylabel("weights")
        for i in range(5):
            fig.suptitle(f"Starting in {5 - i} seconds...")
            plt.pause(1)
        for key in self.history:
            lines = [None, None, None, None]
            for idx, it in enumerate(self.history[key]):
                if not idx % 10 or idx == len(self.history[key]) - 1:
                    fig.suptitle(f"{key}, Iteraction: {idx}")
                    _ = [lines[i].pop(0).remove() for i in range(2)
                         if lines[i] is not None]
                    _ = [lines[i].remove()for i in range(2, len(lines))
                         if lines[i] is not None]
                    lines[0] = axs[0].plot(
                        it['z'], it['h'], color=colors[key], alpha=.5)
                    axs[0].legend(labels=colors.keys())
                    lines[1] = axs[1].plot(
                        costs[key][:idx], color=colors[key], alpha=.5)
                    axs[1].legend(labels=colors.keys())
                    lines[2] = axs[2].scatter(
                        [i for i in range(len(it['gradients']))],
                        it['gradients'], color=colors[key], alpha=.5)
                    axs[2].legend(labels=colors.keys())
                    lines[3] = axs[3].scatter(
                        [i for i in range(len(it['weights']))],
                        it['weights'], color=colors[key], alpha=.5)
                    axs[3].legend(labels=colors.keys())
                    plt.pause(.1)
        plt.show()
