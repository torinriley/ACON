import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.base import clone
from random import randint, uniform


class HyperparameterTuner:
    def __init__(self, model, param_grid, evaluation_metric='accuracy', n_iter=50, exploration_rate=0.1):
        """
        Initialize the HyperparameterTuner.

        :param model: The machine learning model to tune.
        :param param_grid: A dictionary where keys are hyperparameter names and values are lists of possible values.
        :param evaluation_metric: The metric to evaluate the model ('accuracy', 'mse', etc.).
        :param n_iter: Number of iterations for tuning.
        :param exploration_rate: The rate at which new hyperparameters are randomly chosen (exploration).
        """
        self.model = model
        self.param_grid = param_grid
        self.evaluation_metric = evaluation_metric
        self.n_iter = n_iter
        self.exploration_rate = exploration_rate
        self.best_params = {}
        self.best_score = -np.inf if evaluation_metric == 'accuracy' else np.inf

    def evaluate_model(self, X_train, y_train, X_val, y_val, params):
        """
        Train and evaluate the model with given parameters.

        :param X_train: Training features.
        :param y_train: Training labels.
        :param X_val: Validation features.
        :param y_val: Validation labels.
        :param params: Dictionary of hyperparameters to set for the model.
        :return: Evaluation metric score.
        """
        model_clone = clone(self.model)
        model_clone.set_params(**params)
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_val)

        if self.evaluation_metric == 'accuracy':
            score = accuracy_score(y_val, y_pred)
        elif self.evaluation_metric == 'mse':
            score = -mean_squared_error(y_val, y_pred)
        else:
            raise ValueError("Unsupported evaluation metric")

        return score

    def tune(self, X, y):
        """
        Perform adaptive hyperparameter tuning.

        :param X: Features dataset.
        :param y: Labels dataset.
        :return: Best hyperparameters and corresponding evaluation metric score.
        """
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        for i in range(self.n_iter):
            params = {}
            for param in self.param_grid:
                if np.random.rand() < self.exploration_rate:
                    params[param] = np.random.choice(self.param_grid[param])
                else:
                    params[param] = self.best_params.get(param, np.random.choice(self.param_grid[param]))

            score = self.evaluate_model(X_train, y_train, X_val, y_val, params)
            print(f"Iteration {i + 1}/{self.n_iter}, Score: {score}, Params: {params}")

            if (self.evaluation_metric == 'accuracy' and score > self.best_score) or \
                    (self.evaluation_metric == 'mse' and score < self.best_score):
                self.best_score = score
                self.best_params = params

        print(f"Best score: {self.best_score}, Best params: {self.best_params}")
        return self.best_params, self.best_score
