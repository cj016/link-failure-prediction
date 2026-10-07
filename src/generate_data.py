# generate_data.py
# the original live network data is not available, so we make a simulated dataset
# 6000 samples, 19 spans each, 4 raw metrics (as given in the paper)

import numpy as np
import pandas as pd

np.random.seed(1)
n_samples = 6000
n_spans = 19

# target values are fixed for every span
target_span_loss = np.random.uniform(8, 22, n_spans)
target_gain = np.random.uniform(14, 24, n_spans)

# each sample gets an average gain diff (m1) and span loss diff (m2)
m1 = np.zeros(n_samples)
m2 = np.zeros(n_samples)
kind = np.random.rand(n_samples)

for i in range(n_samples):
    if kind[i] < 0.60:          # normal link
        m1[i] = np.random.uniform(-6, 0)
        m2[i] = np.random.uniform(0, 0.5)
    elif kind[i] < 0.80:        # span loss problem
        m1[i] = np.random.uniform(-6, 0)
        m2[i] = np.random.uniform(1, 5.5)
    else:                       # amplifier gain problem
        m1[i] = np.random.uniform(-12, -4)
        m2[i] = np.random.uniform(0, 0.5)

# x1 = amp_gain - target_gain , x2 = target_span_loss - span_loss  (Eq.1)
x1 = m1[:, None] + np.random.normal(0, 0.3, (n_samples, n_spans))
x2 = m2[:, None] + np.random.normal(0, 0.1, (n_samples, n_spans))

amp_gain = target_gain + x1
span_loss = target_span_loss - x2

# OSNR is used for labeling (1 = healthy, 0 = not healthy)
# here osnr is simulated as a linear function of the summed diffs plus noise
X1 = x1.sum(axis=1)
X2 = x2.sum(axis=1)
osnr = 15 + 0.1 * (22 + 0.174 * X1 - 0.14 * X2) + np.random.normal(0, 0.04, n_samples)
label = (osnr >= 15).astype(int)

# build the raw table : span_loss_0..18, target_span_loss_0..18, amp_gain_0..18, target_gain_0..18
data = {}
for s in range(n_spans):
    data["span_loss_" + str(s)] = span_loss[:, s]
for s in range(n_spans):
    data["target_span_loss_" + str(s)] = np.full(n_samples, target_span_loss[s])
for s in range(n_spans):
    data["amp_gain_" + str(s)] = amp_gain[:, s]
for s in range(n_spans):
    data["target_gain_" + str(s)] = np.full(n_samples, target_gain[s])
data["osnr"] = osnr
data["label"] = label

df = pd.DataFrame(data)

# some cells are missing in real data, so remove about 1% randomly
for col in df.columns:
    if col.startswith("span_loss_") or col.startswith("amp_gain_"):
        miss = np.random.rand(n_samples) < 0.01
        df.loc[miss, col] = np.nan

df.round(4).to_csv("data/raw_data.csv", index=False)
print("raw data saved, shape =", df.shape)
print("missing cells =", int(df.isna().sum().sum()))
print("healthy =", int(label.sum()), " not healthy =", int((label == 0).sum()))
