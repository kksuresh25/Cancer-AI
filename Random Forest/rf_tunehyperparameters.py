#import libraries
from sklearn.model_selection import train_test_split
import numpy as np
from imblearn.over_sampling import SMOTE
from svm_constructdata import constructdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

#generate data
features,labels = constructdata('paper_train_data',59)

#seed
seed = 109

#convert label values from -1 to 0
for ii in range(0,len(labels)):
    if (labels[ii] == -1):
        labels[ii] = 0

#split data into test and train
X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size=0.25,random_state=seed)

#Weighted Random Forest Classifier
RF = RandomForestClassifier(random_state=seed,class_weight="balanced")

#Create hyperparameter grid

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

#Find and output results for best parameters based on scoring metric
scores = ['roc_auc','f1']

for score in scores:
    print()
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = RandomizedSearchCV(estimator = RF, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=seed, n_jobs = -1,scoring= score)
    clf.fit(X_train,y_train)
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
