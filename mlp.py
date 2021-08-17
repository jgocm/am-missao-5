import numpy as np
from numpy.core.fromnumeric import argmax, std, var
import os.path
from sklearn.utils.fixes import loguniform

from sklearn.model_selection import RepeatedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

def tune(X_train, X_test, y_train, y_test, search = 'random', n_iter=100):

    # GENERATE CLASSIFIER AND FIND BEST HYPERPARAMETERS
    mlp_search = MLPClassifier(max_iter=5000)
    if (search == 'random'):
        parameter_space = {
            'hidden_layer_sizes': [(10,30,10),(30,30),(7,7)],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha':loguniform(1e-4, 5),
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
        }
        clf = RandomizedSearchCV(mlp_search, parameter_space, n_iter=n_iter, n_jobs=-1, cv=5)
    elif (search == 'grid'):
        parameter_space = {
            'hidden_layer_sizes': [(10,30,10),(30,30),(7,7)],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha':[0.1,0.5,1],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
        }
        clf = GridSearchCV(mlp_search, parameter_space, n_jobs=-1, cv=5)

    clf.fit(X_train, y_train)
    print('Best parameters found:\n', clf.best_params_)

    # GENERATE MODEL
    clf = clf.best_estimator_

    y_true, y_pred = y_test , clf.predict(X_test)
    from sklearn.metrics import classification_report
    print('MLP results on the test set:')
    print(classification_report(y_true, y_pred))
    return clf