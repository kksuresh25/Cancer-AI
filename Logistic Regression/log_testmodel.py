#import libraries
import numpy as np
from imblearn.over_sampling import SMOTE
from svm_constructdata import constructdata
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd

#surpress warnings
import warnings
warnings.filterwarnings("ignore")

#variables for names of datasets
train = 'paper_train_data'
test = 'paper_test_data'

#extract+organize faeatures and labels from training and test datasets
features_train, labels_train = constructdata(train,59)
features_test, labels_test = constructdata(test,59)

#seed
seed = 109

#SMOTE algorithm
sm = SMOTE(random_state=seed, ratio = 1.0)
x_train_res, y_train_res = sm.fit_sample(features_train, labels_train)

#Create the hyptertuned logistic regression classifer
clf = LogisticRegression(C=100,multi_class='multinomial',solver='newton-cg')

#Train the model using the training sets
clf.fit(x_train_res, y_train_res)

#Predict the response for test dataset
y_pred = clf.predict(features_test)

#probability values for ROC AUC score
probs = clf.predict_proba(features_test)
preds = probs[:,1]

#Print scores
print("Accuracy: ",metrics.accuracy_score(labels_test, y_pred))
print("Balanced Accuracy: ",metrics.balanced_accuracy_score(labels_test,y_pred))
print("Precision: ",metrics.precision_score(labels_test, y_pred))
print("Recall: ",metrics.recall_score(labels_test, y_pred))
print("ROC AUC: ",metrics.roc_auc_score(labels_test, preds))
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
