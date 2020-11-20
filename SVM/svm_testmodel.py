#import libraries
from sklearn import svm, metrics
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from svm_constructdata import constructdata

#variables for names of datasets
train = 'paper_train_data'
test = 'paper_test_data'

#extract+organize faeatures and labels from training and test datasets
features_train, labels_train = constructdata(train,59)
features_test, labels_test = constructdata(test,59)

#SMOTE algorithm: Oversample minority in training data
sm = SMOTE(random_state=12, ratio = 1.0)
x_train_res, y_train_res = sm.fit_sample(features_train, labels_train)

#Create the hyptertuned svm classifer
clf = svm.SVC(C = 5, kernel = 'rbf', gamma = .1, probability=True)

#Train the model using the training sets
clf.fit(x_train_res, y_train_res)

#set variables for feautures and labels from test set
x_test = features_test
y_test = labels_test

#Predict the response for training dataset
y_pred = clf.predict(x_test)

#probability values for ROC AUC score
probs = clf.predict_proba(x_test)
preds = probs[:,1]

#Print scores
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
print("Balanced Accuracy: ",metrics.balanced_accuracy_score(y_test,y_pred))
print("Precision: ",metrics.precision_score(y_test, y_pred))
print("Recall: ",metrics.recall_score(y_test, y_pred))
print("ROC AUC: ",metrics.roc_auc_score(y_test, preds))
print()

#output predictions
genes = []
mutations = []
predictions = []

with open(test) as f:
    lines = f.readlines()
for i in range(0,len(lines)):
    words = lines[i].split()
    genes.append(words[0])
    mutations.append(words[1]+words[2]+words[3])
    if y_pred[i] == 0:
        predictions.append(-1)
    else:
        predictions.append(y_pred[i])

df = pd.DataFrame({'Gene': genes})
df.insert(1,'Mutation',mutations)
df.insert(2,'Activating Status Prediction',predictions)
df.to_csv('ai_results.csv',index=False,header=True)
