# gda.py
# Gaussian Discriminant Analysis (Eq.4) - same covariance matrix for both classes

import numpy as np


class GDA:
    def fit(self, X, y):
        n = X.shape[0]
        self.phi = np.mean(y)                      # p(y=1) = phi
        self.mu0 = X[y == 0].mean(axis=0)
        self.mu1 = X[y == 1].mean(axis=0)

        # shared covariance matrix sigma
        centered = X - np.where(y[:, None] == 1, self.mu1, self.mu0)
        self.sigma = centered.T @ centered / n
        self.sigma_inv = np.linalg.inv(self.sigma)

        # linear form of the boundary : theta0 + theta1*x1 + theta2*x2 = 0
        self.theta = self.sigma_inv @ (self.mu1 - self.mu0)
        self.theta0 = (0.5 * (self.mu0 @ self.sigma_inv @ self.mu0
                              - self.mu1 @ self.sigma_inv @ self.mu1)
                       + np.log(self.phi / (1 - self.phi)))
        return self

    def log_gaussian(self, X, mu):
        # log of p(x|y) from Eq.4
        d = X.shape[1]
        diff = X - mu
        quad = np.sum((diff @ self.sigma_inv) * diff, axis=1)
        return (-0.5 * d * np.log(2 * np.pi)
                - 0.5 * np.log(np.linalg.det(self.sigma))
                - 0.5 * quad)

    def predict(self, X):
        # pick the class with bigger p(x|y) * p(y)
        score1 = self.log_gaussian(X, self.mu1) + np.log(self.phi)
        score0 = self.log_gaussian(X, self.mu0) + np.log(1 - self.phi)
        return (score1 > score0).astype(int)
