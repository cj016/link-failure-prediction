# feature_engineering.py
# fills missing values, then makes the two features X1 and X2 (Eq.1 and Eq.2)

import numpy as np
import pandas as pd

df = pd.read_csv("data/raw_data.csv")
n_spans = 19

span_loss_cols = ["span_loss_" + str(s) for s in range(n_spans)]
target_loss_cols = ["target_span_loss_" + str(s) for s in range(n_spans)]
amp_gain_cols = ["amp_gain_" + str(s) for s in range(n_spans)]
target_gain_cols = ["target_gain_" + str(s) for s in range(n_spans)]

# missing values -> linear interpolation between neighbour cells (across the spans)
print("missing before =", int(df[span_loss_cols + amp_gain_cols].isna().sum().sum()))
df[span_loss_cols] = df[span_loss_cols].interpolate(axis=1, limit_direction="both")
df[amp_gain_cols] = df[amp_gain_cols].interpolate(axis=1, limit_direction="both")
print("missing after  =", int(df[span_loss_cols + amp_gain_cols].isna().sum().sum()))

# Eq.1 : x1 = amp_gain - target_gain , x2 = target_span_loss - span_loss
x1 = df[amp_gain_cols].values - df[target_gain_cols].values
x2 = df[target_loss_cols].values - df[span_loss_cols].values

# Eq.2 : X1 = sum of x1 over the 19 spans (n = 0 to 18), same for X2
X1 = x1.sum(axis=1)
X2 = x2.sum(axis=1)

features = pd.DataFrame({"X1": X1, "X2": X2, "y": df["label"]})
features.to_csv("data/features.csv", index=False)

print("features saved, shape =", features.shape)
print(features.describe().round(2))
