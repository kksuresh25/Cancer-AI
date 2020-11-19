#import libraries
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from svm_constructdata import constructdata
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

#generate data
features_train, labels_train = constructdata('paper_train_data',59)

#seed
seed = 109

#SMOTE algorithm
sm = SMOTE(random_state=seed, ratio = 1.0)
x_train_res, y_train_res = sm.fit_sample(features_train, labels_train)

#Set the parameters by cross-validation
tuned_parameters = [{'solver': ['newton-cg', 'lbfgs', 'sag'],
                    'multi_class': ['ovr', 'multinomial'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    }]

#Find and output results for best parameters based on scoring metric
#Other metrics: 'accuracy','recall','precision','f1'
scores = ['roc_auc']

for score in scores:
    print()
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = GridSearchCV(LogisticRegression(), tuned_parameters, cv=5,
                       scoring= score)
    clf.fit(x_train_res,y_train_res)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
