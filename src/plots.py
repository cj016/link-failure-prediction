# plots.py
# all the figures for the project

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_boundary(X, y, theta0, theta1, theta2, title, filename):
    # black dots = not healthy (0), green x = healthy (1), blue line is p(y|x) = 0.5
    plt.figure(figsize=(6, 5))
    plt.plot(X[y == 0, 0], X[y == 0, 1], "k.", label="not healthy (y=0)")
    plt.plot(X[y == 1, 0], X[y == 1, 1], "gx", label="healthy (y=1)")

    x1_line = np.array([X[:, 0].min(), X[:, 0].max()])
    x2_line = -(theta0 + theta1 * x1_line) / theta2
    plt.plot(x1_line, x2_line, "b-", label="decision boundary")

    plt.ylim(X[:, 1].min() - 5, X[:, 1].max() + 5)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_accuracy_table(names, accs, filename):
    # table like Fig.3 in the paper
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.axis("off")
    rows = [[n, str(round(a * 100, 2)) + "%"] for n, a in zip(names, accs)]
    table = ax.table(cellText=rows, colLabels=["Classifier", "Test Accuracy"],
                     loc="center", cellLoc="center")
    table.scale(1, 1.6)
    plt.title("Test accuracy of the classifiers")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cost(costs, filename):
    # cost J(theta) of logistic regression at every newton iteration
    plt.figure(figsize=(5, 4))
    plt.plot(range(len(costs)), costs, "o-")
    plt.xlabel("iteration")
    plt.ylabel("cost J(theta)")
    plt.title("Newton's method - cost vs iteration")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
