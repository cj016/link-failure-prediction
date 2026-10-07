# svm_model.py
# SVM with linear kernel

import numpy as np
from sklearn.svm import SVC


def linear_kernel(a, b):
    # K(x, y) = phi(x)^T phi(y) with phi(x) = x for the linear kernel (Eq.5)
    return a @ b.T


def train_svm(X_train, y_train):
    model = SVC(kernel="linear", C=1.0)
    model.fit(X_train, y_train)
    return model


def check_with_kernel(model, X):
    # decision value = sum(alpha_i * y_i * K(sv_i, x)) + b  (dual form)
    k = linear_kernel(model.support_vectors_, X)
    manual = model.dual_coef_ @ k + model.intercept_
    return np.allclose(manual.ravel(), model.decision_function(X))
