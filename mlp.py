import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax
import pandas as pd
import os.path

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm
import data_analysis

# IMPORT DATASET WITH PANDAS
DF_PATH = os.path.dirname(__file__) + "/glass.csv"
df = pd.read_csv(DF_PATH, encoding="utf-8")
X = np.array(df.iloc[:, :-1])
y = np.array(df.iloc[:, -1:])

# DATA EXPLORATORY ANALYSIS AND PREPROCESSING
# data_analysis.run(df, False)

df_drop = df.drop(columns="RI")
# print(df_drop.info())

# ADJUST HYPERPARAMETERS WITH VALIDATION SET
# SPLIT DATASET
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.8, random_state=42
)

# STANDARDIZATION
from sklearn.preprocessing import StandardScaler

scaler1 = StandardScaler()
# Fit only on training data
scaler1.fit(X_train)
X_train_std = scaler1.transform(X_train)
# apply same transformation to test data
X_test_std = scaler1.transform(X_test)

# NORMALIZATION
from sklearn.preprocessing import Normalizer

scaler2 = Normalizer()
# Fit only on training data
scaler2.fit(X_train)
X_train_norm = scaler2.transform(X_train)
# apply same transformation to test data
X_test_norm = scaler2.transform(X_test)

# Feature scaler
scaler = "std"

if scaler == "std":
    X_train = X_train_std
    X_test = X_test_std
if scaler == "norm":
    X_train = X_train_norm
    X_test = X_test_norm

# GENERATE CLASSIFIER AND FIND BEST HYPERPARAMETERS
mlp_gs = MLPClassifier(max_iter=10000)
parameter_space = {
    "hidden_layer_sizes": [(10, 30, 10), (20,), (30, 30), (100, 30), (30, 100)],
    "activation": ["identity", "logistic", "tanh", "relu"],
    "solver": ["lbfgs", "sgd", "adam"],
    "alpha": [0.0001, 0.05, 1, 3, 0.5],
    "learning_rate": ["constant", "invscaling", "adaptive"],
    "batch_size": [10, 20, 50],
}

clf = RandomizedSearchCV(
    mlp_gs, parameter_space, n_iter=300, n_jobs=-1, cv=5, return_train_score=True
)
clf.fit(X_train, y_train)  # X is train samples and y is the corresponding labels

print("Best parameters found:\n", clf.best_params_)

means = clf.cv_results_["mean_test_score"]
stds = clf.cv_results_["std_test_score"]
for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

y_true, y_pred = y_test, clf.predict(X_test)
from sklearn.metrics import classification_report

print("Results on the test set:")
print(classification_report(y_true, y_pred))

test_scores = clf.cv_results_["mean_test_score"]
train_scores = clf.cv_results_["mean_train_score"]
plt.plot(test_scores, label="test")
plt.plot(train_scores, label="train")
plt.legend(loc="best")
plt.show()

# GENERATE MODELS


# 10 REPETITIONS OF 5-FOLD CROSS VALIDATION
n_splits = 5
n_repeats = 10

# TEST MODEL
scores = []
rkf = RepeatedKFold(n_splits, n_repeats)
for train_index, test_index in rkf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # scores.append(clf.fit(X_train, y_train).score(X_test, y_test))
# print(scores)
