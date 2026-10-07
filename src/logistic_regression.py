# logistic_regression.py
# logistic regression trained with Newton's method (Eq.3)

import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))          # g(z) = 1 / (1 + e^-z)


class LogisticRegressionNewton:
    def __init__(self, max_iter=25, tol=1e-8):
        self.max_iter = max_iter
        self.tol = tol
        self.theta = None
        self.costs = []

    def cost(self, Xb, y):
        h = sigmoid(Xb @ self.theta)
        h = np.clip(h, 1e-12, 1 - 1e-12)
        # J(theta) = -1/n * sum( y log(h) + (1-y) log(1-h) )
        return -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))

    def fit(self, X, y):
        n = X.shape[0]
        Xb = np.c_[np.ones(n), X]            # add the theta0 column
        self.theta = np.zeros(Xb.shape[1])
        self.costs = [self.cost(Xb, y)]

        for i in range(self.max_iter):
            h = sigmoid(Xb @ self.theta)
            grad = Xb.T @ (h - y) / n                    # gradient
            H = (Xb.T * (h * (1 - h))) @ Xb / n          # hessian
            step = np.linalg.solve(H, grad)              # H^-1 * grad
            self.theta = self.theta - step               # newton update
            self.costs.append(self.cost(Xb, y))
            if np.abs(step).max() < self.tol:
                break
        return self

    def predict_proba(self, X):
        Xb = np.c_[np.ones(X.shape[0]), X]
        return sigmoid(Xb @ self.theta)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)   # p(y|x) = 0.5 boundary
