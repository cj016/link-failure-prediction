# Link Failure Prediction in Cloud Scale Networks using Supervised Learning

UE24CS352A - Machine Learning mini project

Team:
- Chetan Jevoor - PES2UG24CS132
- Harkirat Singh - PES2UG24CS180

Based on the paper "Link Failure prediction and Localization in Cloud Scale Networks using Supervised Learning" (Zahra Bakhtiari).

## Problem statement
Predict whether an optical link in a cloud network is healthy or not healthy, using supervised learning classifiers, so that failures can be handled before customers are affected.

## Dataset
The paper used live network data which is not available to us, so we **simulated a dataset** with the same structure as described in the paper:
- 6000 samples, each with 19 spans
- 4 raw metrics per span: span loss, target span loss, amplifier gain, target gain
- about 1% of the cells are made missing on purpose and filled by linear interpolation
- label (1 = healthy, 0 = not healthy) is decided from a simulated OSNR value

Features (Eq.1 and Eq.2 of the paper):
- x1 = amp_gain - target_gain, x2 = target_span_loss - span_loss (per span)
- X1 = sum of x1 over the 19 spans, X2 = sum of x2 over the 19 spans

## Approach
- 75% data for training (4500 points) and 25% for testing (1500 points)
- Logistic regression trained with Newton's method (cost from Eq.3)
- Gaussian Discriminant Analysis (Eq.4)
- SVM with linear kernel (Eq.5)
- Test accuracy as in Eq.6, plus precision and recall

## Folder structure
```
src/generate_data.py        makes the simulated raw data
src/feature_engineering.py  missing values + X1, X2
src/logistic_regression.py  logistic regression with Newton's method
src/gda.py                  GDA
src/svm_model.py            linear SVM
src/evaluate.py             accuracy, precision, recall
src/plots.py                all the figures
src/main.py                 runs everything
data/                       raw_data.csv and features.csv
results/                    figures and csv results
```

## How to run
Python 3.9 or above is needed. Run all commands from the project folder.

```
pip install -r requirements.txt
python src/generate_data.py
python src/feature_engineering.py
python src/main.py
```
(use `python3` instead of `python` on Mac/Linux)

The figures and result files are saved in the `results/` folder.

## Results (test data, 1500 points)
| Classifier | Test accuracy | Paper |
|---|---|---|
| Logistic Regression | 99.27% | 99% |
| GDA | 98.13% | 97% |
| Linear SVM | 99.33% | 99% |

Model coefficients are in `results/coefficients.csv` (like Table 1 of the paper).
Since our data is simulated the coefficients are not the same as the paper, but the linear classifiers separate the two classes well in the same way.

## Contribution
- Chetan: data generation, feature engineering, logistic regression (Newton's method), main script, README
- Harkirat: GDA, linear SVM, evaluation metrics, plots and results
