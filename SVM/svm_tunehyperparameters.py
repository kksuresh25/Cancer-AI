#import libraries
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from svm_constructdata import constructdata

#variables for features and labels from training data
features_train, labels_train = constructdata('paper_train_data',59)

#Oversample minority in training data

#SMOTE algorithm
sm = SMOTE(random_state=12, ratio = 1.0)
x_train_res, y_train_res = sm.fit_sample(features_train, labels_train)

#Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'],
                    'C': [.01,.1,1,2,3,4,5],
                    'gamma': [0.00001, .0001, .001, .01, .1, 1,
                                10, 100, 1000, 10000]
                     }]


#Find and output results for best parameters based on scoring metric
#Other metrics: 'accuracy','recall','precision','f1'
scores = ['roc_auc']

for score in scores:
    print()
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
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

Cs = [.01,.1,1,2,3,4,5]
Gammas = [0.00001, .0001, .001, .01, .1, 1,10, 100, 1000, 10000]
def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], 'o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores: ROC_AUC", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_xscale('log')
    ax.set_ylabel('CV Average ROC_AUC Score', fontsize=16)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True)
    plt.show()

# Calling Method
plot_grid_search(clf.cv_results_, Gammas, Cs, 'Gamma', 'C')
