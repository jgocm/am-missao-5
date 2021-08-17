import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax

from sklearn import svm
import data_analysis

def tune(X_train, X_test, y_train, y_test):

    # GENERATE CLASSIFIERS AND FIND BEST "C" FOR EACH
    gamma = "auto"
    lin_scores = [svm.SVC(kernel="linear", gamma=gamma, C=c).fit(X_train, y_train).score(X_test, y_test) for c in range(1,200)]
    best_lin = [max(lin_scores), argmax(lin_scores)+1]
    rbf_scores = [svm.SVC(kernel="rbf", gamma=gamma, C=c).fit(X_train, y_train).score(X_test, y_test) for c in range(1,200)]
    best_rbf = [max(rbf_scores), argmax(rbf_scores)+1]
    sgd_scores = [svm.SVC(kernel="sigmoid", gamma=gamma, C=c).fit(X_train, y_train).score(X_test, y_test) for c in range(1,200)]
    best_sgd = [max(sgd_scores), argmax(sgd_scores)+1]
    poly_scores = []
    for degree in range (3):
        current_scores=([svm.SVC(kernel="poly", gamma=gamma, degree=degree, C=c).fit(X_train, y_train).score(X_test, y_test) for c in range(1,200)])
        poly_scores.append([max(current_scores),argmax(current_scores)+1, degree])
    poly_scores=np.array(poly_scores)
    poly_scores = [svm.SVC(kernel="poly", gamma=gamma, degree=3, C=c).fit(X_train, y_train).score(X_test, y_test) for c in range(1,200)]
    best_poly=[max(poly_scores), argmax(poly_scores)+1]
    x_plot=range(1,200)
    plt.ylabel("accuracy")
    plt.xlabel("C")
    plt.title("C validation")
    plt.plot(x_plot,lin_scores,'r', label="linear")
    plt.plot(x_plot,rbf_scores,'b', label="rbf")
    plt.plot(x_plot,sgd_scores,'g', label="sigmoid")
    plt.plot(x_plot,poly_scores,'y', label="polynomial")
    plt.legend()
    plt.show()

    # GENERATE MODELS
    C = best_lin[1]
    #C=147
    svm_lin = svm.SVC(kernel="linear", gamma=gamma, C=C)
    C = best_rbf[1]
    #C=190
    svm_rbf = svm.SVC(kernel="rbf", gamma=gamma, C=C)
    C = best_sgd[1]
    #C=171
    svm_sgd = svm.SVC(kernel="sigmoid", gamma=gamma, C=C)
    C = best_poly[1]
    #C=159
    degree = 3
    svm_poly = svm.SVC(kernel="poly", gamma=gamma, C=C, degree=degree)

    y_true, y_pred = y_test, svm_lin.fit(X_train, y_train).predict(X_test)
    from sklearn.metrics import classification_report
    print('Linear SVM results on the test set:')
    print(classification_report(y_true, y_pred))

    y_true, y_pred = y_test, svm_rbf.fit(X_train, y_train).predict(X_test)
    from sklearn.metrics import classification_report
    print('RBF SVM results on the test set:')
    print(classification_report(y_true, y_pred))

    y_true, y_pred = y_test, svm_sgd.fit(X_train, y_train).predict(X_test)
    from sklearn.metrics import classification_report
    print('Sigmoid SVM results on the test set:')
    print(classification_report(y_true, y_pred))

    y_true, y_pred = y_test, svm_poly.fit(X_train, y_train).predict(X_test)
    from sklearn.metrics import classification_report
    print('Polynomial SVM results on the test set:')
    print(classification_report(y_true, y_pred))

    return svm_lin, svm_rbf, svm_sgd, svm_poly
