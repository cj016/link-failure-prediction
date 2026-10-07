# main.py
# trains logistic regression, GDA and linear SVM and compares them
# run from the project folder :  python src/main.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from logistic_regression import LogisticRegressionNewton
from gda import GDA
from svm_model import train_svm, check_with_kernel
from evaluate import test_accuracy, precision_recall, confusion
import plots

if not os.path.exists("data/features.csv"):
    print("features.csv not found, run generate_data.py and feature_engineering.py first")
    raise SystemExit

os.makedirs("results", exist_ok=True)

# load the features X1, X2 and label y
data = pd.read_csv("data/features.csv")
X = data[["X1", "X2"]].values
y = data["y"].values
print("total data points =", len(data))

# 75% train, 25% test  (4500 / 1500)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("train =", len(X_train), " test =", len(X_test))

# 1. logistic regression (newton's method)
lr = LogisticRegressionNewton()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print("\nLogistic regression")
print("theta =", lr.theta.round(4))
print("final cost =", round(lr.costs[-1], 5), " iterations =", len(lr.costs) - 1)

# 2. GDA
gda = GDA()
gda.fit(X_train, y_train)
gda_pred = gda.predict(X_test)
print("\nGDA")
print("phi =", round(gda.phi, 4))
print("mu0 =", gda.mu0.round(3), " mu1 =", gda.mu1.round(3))
print("sigma =\n", gda.sigma.round(3))
print("theta =", np.array([gda.theta0, gda.theta[0], gda.theta[1]]).round(4))

# 3. SVM with linear kernel
svm = train_svm(X_train, y_train)
svm_pred = svm.predict(X_test)
print("\nLinear SVM")
print("w =", svm.coef_[0].round(4), " b =", svm.intercept_.round(4))
print("support vectors =", len(svm.support_vectors_))
print("dual form check with K(x,y) :", check_with_kernel(svm, X_test))

# test accuracy (Eq.6)
names = ["Logistic Regression", "GDA", "Linear SVM"]
preds = [lr_pred, gda_pred, svm_pred]
accs = [test_accuracy(y_test, p) for p in preds]

print("\nResults on test data")
print("%-20s %-10s %-10s %-10s" % ("Classifier", "Accuracy", "Precision", "Recall"))
rows = []
for name, p, a in zip(names, preds, accs):
    prec, rec = precision_recall(y_test, p)
    print("%-20s %-10.4f %-10.4f %-10.4f" % (name, a, prec, rec))
    tp, tn, fp, fn = confusion(y_test, p)
    rows.append([name, round(a, 4), round(prec, 4), round(rec, 4), tp, tn, fp, fn])
    print("   confusion -> TP:%d TN:%d FP:%d FN:%d" % (tp, tn, fp, fn))

pd.DataFrame(rows, columns=["classifier", "accuracy", "precision", "recall",
                            "TP", "TN", "FP", "FN"]).to_csv("results/results.csv", index=False)

# model coefficients (Table 1 in the paper)
coef = pd.DataFrame({
    "classifier": ["GDA", "Logistic Regression"],
    "theta0": [gda.theta0, lr.theta[0]],
    "theta1": [gda.theta[0], lr.theta[1]],
    "theta2": [gda.theta[1], lr.theta[2]],
}).round(4)
print("\nModel coefficients (Table 1)")
print(coef.to_string(index=False))
coef.to_csv("results/coefficients.csv", index=False)

# figures
plots.plot_accuracy_table(names, accs, "results/fig3_accuracy_table.png")
plots.plot_boundary(X_test, y_test, lr.theta[0], lr.theta[1], lr.theta[2],
                    "Logistic regression", "results/fig4_logistic_regression.png")
plots.plot_boundary(X_test, y_test, gda.theta0, gda.theta[0], gda.theta[1],
                    "GDA", "results/fig5_gda.png")
plots.plot_boundary(X_test, y_test, svm.intercept_[0], svm.coef_[0][0], svm.coef_[0][1],
                    "SVM (linear kernel)", "results/fig6_linear_svm.png")
plots.plot_cost(lr.costs, "results/fig7_newton_cost.png")
print("\nfigures saved in results/")
