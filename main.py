from inspect import stack
from scipy.sparse import data
import numpy as np
import knn, svm, mlp, data_analysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os.path

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier

import warnings
warnings.filterwarnings('ignore')
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

DF_PATH = os.path.dirname(__file__) + "/glass.csv"
df = pd.read_csv(DF_PATH, encoding="utf-8")

X = np.array(df.iloc[:,  :-1])
y = np.array(df.iloc[:,-1:  ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

#data_analysis.run(df)

# STANDARDIZATION
std_scaler = StandardScaler()  
# Fit only on training data
std_scaler.fit(X_train)  
X_train_std = std_scaler.transform(X_train)  
# apply same transformation to test data
X_test_std = std_scaler.transform(X_test)

X_train = X_train_std
X_test = X_test_std

svm_lin, svm_rbf, svm_sgd, svm_poly = svm.tune(X_train, X_test, y_train, y_test)
mlp_clf = mlp.tune(X_train, X_test, y_train, y_test, search='random', n_iter=50)
knn_clf = knn.tune(X_train, X_test, y_train, y_test)

def bagging(estimator):
    bag = BaggingClassifier(base_estimator=estimator,
                            n_estimators=6,
                            max_samples=1.0).fit(X_train, y_train)

    y_true, y_pred = y_test , bag.predict(X_test)
    print(f'{bag.base_estimator_} bagging results on the test set:')
    print(classification_report(y_true, y_pred))
    return bag

def gradBoosting(estimator):
    boost = GradientBoostingClassifier(init=estimator,
                            n_estimators=100).fit(X_train, y_train)

    y_true, y_pred = y_test , boost.predict(X_test)
    print(f'{boost.init} gradient boost results on the test set:')
    print(classification_report(y_true, y_pred))
    return boost

def adaBoosting(estimator):
    boost = AdaBoostClassifier(base_estimator=estimator,
                            n_estimators=100).fit(X_train, y_train)

    y_true, y_pred = y_test , boost.predict(X_test)
    print(f'{boost.base_estimator_} AdaBoost results on the test set:')
    print(classification_report(y_true, y_pred))
    return boost

def stacking(estimators):
    stack = StackingClassifier(estimators=estimators,
                            cv=5).fit(X_train, y_train)

    y_true, y_pred = y_test , stack.predict(X_test)
    print(f'{stack.estimators} stacking results on the test set:')
    print(classification_report(y_true, y_pred))
    return stack

mlp_bag = bagging(mlp_clf)
knn_bag = bagging(knn_clf)
svm_bag = bagging(svm_rbf)


mlp_boost = gradBoosting(mlp_clf)
knn_boost = gradBoosting(knn_clf)
#svm_boost = adaBoosting(svm_poly)

stack = stacking(estimators=[('MLP',mlp_clf), ('KNN',knn_clf),('SVM',svm_rbf)])
