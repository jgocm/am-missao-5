import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier

def tune(X_train, X_test, y_train, y_test):

    k_range=range(1,20)
    train_scores = [KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train).score(X_train, y_train) for n_neighbors in k_range]
    test_scores = [KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train).score(X_test, y_test) for n_neighbors in k_range]
    best_knn = [max(test_scores), argmax(test_scores)+1]

    plt.ylabel("accuracy")
    plt.xlabel("K")
    plt.title("K validation")
    plt.plot(k_range,test_scores,'r')
    plt.plot(k_range,train_scores,'b')
    plt.show()

    clf = KNeighborsClassifier(n_neighbors=best_knn[1])
    y_true, y_pred = y_test , clf.fit(X_train, y_train).predict(X_test)
    from sklearn.metrics import classification_report
    print('KNN results on the test set:')
    print(classification_report(y_true, y_pred))
    return clf