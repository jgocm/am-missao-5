import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax
import pandas as pd
import os.path

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn import svm
import data_analysis

# IMPORT DATASET WITH PANDAS
DF_PATH = os.path.dirname(__file__) + "/glass.csv"
df = pd.read_csv(DF_PATH, encoding="utf-8")
X = np.array(df.iloc[:,  :-1])
y = np.array(df.iloc[:,-1:  ])

# DATA EXPLORATORY ANALYSIS AND PREPROCESSING
#data_analysis.run(df, False)

df_std=data_analysis.attributes_standardization(df)
df_norm=data_analysis.attributes_normalization(df)
df_drop = df.drop(columns="RI")
#print(df_drop.info())

# ADJUST HYPERPARAMETERS WITH VALIDATION SET
# SPLIT DATASET
X = np.array(df.iloc[:,  :-1])
y = np.array(df.iloc[:,-1:  ])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# GENERATE CLASSIFIERS AND FIND BEST "C" FOR EACH
# BEST CLASSIFIER: best_clf = [accuracy C]
# FOR POLYNOMIAL: best_poly = [accuracy C degree]
gamma = "auto"
lin_scores = [svm.SVC(kernel="linear", gamma=gamma, C=c).fit(X_train, y_train).score(X_test, y_test) for c in range(1,10)]
best_lin = [max(lin_scores), argmax(lin_scores)+1]
rbf_scores = [svm.SVC(kernel="rbf", gamma=gamma, C=c).fit(X_train, y_train).score(X_test, y_test) for c in range(1,10)]
best_rbf = [max(rbf_scores), argmax(rbf_scores)+1]
sgd_scores = [svm.SVC(kernel="sigmoid", gamma=gamma, C=c).fit(X_train, y_train).score(X_test, y_test) for c in range(1,10)]
best_sgd = [max(sgd_scores), argmax(sgd_scores)+1]
poly_scores = []
for degree in range (2,20):
    current_scores=([svm.SVC(kernel="poly", gamma=gamma, degree=degree, C=c).fit(X_train, y_train).score(X_test, y_test) for c in range(1,10)])
    poly_scores.append([max(current_scores),argmax(current_scores)+1, degree])
poly_scores=np.array(poly_scores)
x_plot=poly_scores[:,2]
scores=poly_scores[:,0]
y_plot=scores
best_poly=poly_scores[argmax(scores)]
plt.ylabel("accuracy")
plt.xlabel("degree")
plt.title("Polynomial Kernels")
plt.grid(1)
plt.xticks(x_plot)
plt.plot(x_plot,y_plot) 
plt.show()

# GENERATE MODELS
#C = best_lin[1]
C=3
svm_lin = svm.SVC(kernel="linear", gamma=gamma, C=C)
#C = best_rbf[1]
C=7
svm_rbf = svm.SVC(kernel="rbf", gamma=gamma, C=C)
#C = best_sgd[1]
C=1
svm_sgd = svm.SVC(kernel="sigmoid", gamma=gamma, C=C)
#C = best_poly[1]
C=1
degree = 5
svm_poly = svm.SVC(kernel="poly", gamma=gamma, C=C, degree=degree)

# 10 REPETITIONS OF 5-FOLD CROSS VALIDATION
n_splits = 5
n_repeats = 10

# TEST MODEL
scores = []
rkf = RepeatedKFold(n_splits, n_repeats)
for train_index, test_index in rkf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    scores.append([clf.fit(X_train, y_train).score(X_test, y_test) for clf in (svm_lin, svm_rbf, svm_sgd, svm_poly)])
print(scores)
lin_scores=np.array(scores[:,0])
rbf_scores=np.array(scores[:,1])
sgd_scores=np.array(scores[:,2])
svm_scores=np.array(scores[:,3])
print(svm_lin.C, svm_rbf.C, svm_sgd.C, svm_poly.C)