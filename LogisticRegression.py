import json
import math
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame

class LogisticRegression:
    DEFAULT_MAX_EPOCHS=300
    DEFAULT_LEARNING_RATE = .15
    DEFAULT_DELTA_THRESHOLD = 1e-7

    def __init__(self, verbose=False, fit_intercept=True):
        """
        initializes a logistic regression model

        Args:
            verbose (bool, optional): Displays log. Defaults to False.
            fit_intercept (bool, optional): calculates biases. Defaults to True.
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

    def __log(self, *values: object):
        """
        prints if self.verbose is true.
        """
        if self.verbose:
            print(*values)

    def __sigmoid(self, z: float):
        """
        Computes the sigmoid function for a given input

        Args:
            z (float): _description_

        Returns:
            float: sigmoid of the input valud
        """
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
        sets its value to 1 if its label value == clas else, to zero
        """
        y_class = np.where(y == clas, 1, 0)
        y_class = y_class.astype(np.int64)
        return (y_class)


    def __normalize(self, X, min_values=None, max_values=None):
        self.min_values = np.min(X, axis=0) if min_values is None else min_values
        self.max_values = np.max(X, axis=0) if max_values is None else max_values
        X_scaled = (X - self.min_values) / (self.max_values - self.min_values)
        return X_scaled


    def __gradient_descent(self, X, y):
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
            if self.graph == True:
                history.append({'z': np.sort(z).tolist(), 'h': np.sort(h).tolist(), 'bias': bias.tolist(), 'weights': thetas.tolist(), 'cost': cost, 'gradients': gradients.tolist()})
        return(thetas, bias, history)


    def __train(self):
        """
        for each class, create a one_vs_all and get the weights of its features
        using a gradient descent
        """
        self.weights = {}
        self.biases = {}
        self.history = {}
        for c in self.classes:
            self.__log("Training for class:", c)
            self.weights[c] = {}
            y_c = self.__one_vs_all(self.y, c)
            weights, bias, history = self.__gradient_descent(self.X, y_c) #* ratio
            self.weights[c] = weights
            self.biases[c] = bias
            self.history[c] = history
        self.__log("Weights:", self.weights)
        self.__log("Biases:", self.biases)


    def fit(self, X, y, normalize=True, delta_threshold=DEFAULT_DELTA_THRESHOLD, max_iterations=DEFAULT_MAX_EPOCHS, learning_rate=DEFAULT_LEARNING_RATE, graph=True):
        self.epochs = max_iterations
        self.learning_rate = learning_rate
        self.delta_threshold = delta_threshold
        self.X = X
        self.y = y
        self.classes = np.unique(self.y)
        self.classes.sort()
        if normalize:
            self.X = self.__normalize(self.X)
        if graph:
            self.graph = True
        self.__train()



    def save_model(self, filename='model.json'):
        dic = {}
        dic['weights'] = None if self.weights is None else {k: self.weights[k].tolist() for k in self.weights.keys()}
        dic['biases'] = None if self.biases is None else {k: self.biases[k].tolist() for k in self.biases.keys()}
        dic['classes'] = None if self.classes is None else self.classes.tolist()
        #SAVING HISTORY JUST FOR TESTING RIGHT NOW
        dic['history'] = self.history
        dic['min_values'] = None if self.min_values is None else self.min_values.tolist()
        dic['max_values'] = None if self.max_values is None else self.max_values.tolist()
        with open(filename, "w") as outfile:
            outfile.write(json.dumps(dic, indent=4))
    
    
    def load_model(self, filename='model.json'):
        dic = {}
        with open(filename, 'r') as infile:
            dic = json.load(infile)        
        self.weights = {k: np.array(dic['weights'][k]) for k in dic['weights'].keys()} if 'weights' in dic else None
        self.biases = {k: np.array(dic['biases'][k]) for k in dic['biases'].keys()} if 'biases' in dic else None
        self.classes = np.array(dic['classes']) if 'classes' in dic else None
        self.min_values = np.array(dic['min_values']) if 'min_values' in dic else None
        self.max_values = np.array(dic['max_values']) if 'max_values' in dic else None
        # LOADING HISTORY JUST FOR TESTING
        self.history = dic['history']

        

    def score(self, X, y):
        ds = self.predict(X, labelname = 'prediction')
        ds['true'] = y
        print(ds)
        times_right = (ds['prediction'] == ds['true']).sum()
        result = times_right / len(ds)
        return (result)
    

    def predict(self, X, normalize=True, save=True, filename='predictions.csv', labelname='Predictions'):
        ds = DataFrame()
        if normalize == True:
            X = self.__normalize(X, self.min_values, self.max_values)
        for c in self.weights:
            print(c, self.weights[c])
            linear_pred = np.dot(X, self.weights[c]) + self.biases[c]
            ds[c] = self.__sigmoid(linear_pred)
        ds[labelname] = ds[self.weights.keys()].idxmax(axis=1)
        ds = ds[[labelname]]
        self.__log(ds)
        if save:
            ds.to_csv(filename, index_label='Index')
        return(ds)


    def graph(self):
        """
        uses history to graph.
        """
        if not self.history:
            print("Nothing to Graph")
            return
        colorsmap = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        colors = {k: colorsmap[i] for i, k in enumerate(self.history.keys())}
        print(colors)
        fig, axs = plt.subplots(4)
        #group all costs:
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
                if not idx % 10:
                    fig.suptitle(f"{key}, Iteraction: {idx}")
                    _ = [lines[i].pop(0).remove() for i in range(len(lines)) if lines[i] is not None]
                    lines[0] = axs[0].plot(it['z'], it['h'], color=colors[key], alpha=.5)
                    axs[0].legend(labels=colors.keys())
                    lines[1] = axs[1].plot(costs[key][:idx], color=colors[key], alpha=.5)
                    axs[1].legend(labels=colors.keys())
                    lines[2] = axs[2].plot(it['gradients'], color=colors[key], alpha=.5)
                    axs[2].legend(labels=colors.keys())
                    lines[3] = axs[3].plot(it['weights'], color=colors[key], alpha=.5)
                    axs[3].legend(labels=colors.keys())
                    plt.pause(.1)


        plt.show()
