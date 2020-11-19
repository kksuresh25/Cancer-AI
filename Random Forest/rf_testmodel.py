#import libraries
from sklearn import metrics
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from svm_constructdata import constructdata
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

#variables for names of datasets
train = 'paper_train_data'
test = 'paper_test_data'

#extract+organize faeatures and labels from training and test datasets
features_train, labels_train = constructdata(train,59)
features_test, labels_test = constructdata(test,59)

#seed
seed = 109

#convert label values from -1 to 0
for ii in range(0,len(labels_train)):
    if (labels_train[ii] == -1):
        labels_train[ii] = 0

for ii in range(0,len(labels_test)):
    if (labels_test[ii] == -1):
        labels_test[ii] = 0

# #Upsample minority class
sm = SMOTE(random_state=seed, ratio = 1.0)
features_train, labels_train = sm.fit_sample(features_train, labels_train)

#Weighted Random Forest Classifier
RF = RandomForestClassifier(n_estimators=1200,min_samples_split=10,min_samples_leaf=1, max_features="sqrt",max_depth=None,bootstrap=True,random_state=seed,class_weight="balanced",n_jobs=-1)

#Train classifier
RF.fit(features_train, labels_train)

#Predict on test data
y_pred = RF.predict(features_test)

#Evaluate Results
print("Accuracy: ",metrics.accuracy_score(labels_test, y_pred))
print("Balanced Accuracy: ",metrics.balanced_accuracy_score(labels_test,y_pred))
print("ROC AUC: ",metrics.roc_auc_score(labels_test, y_pred))
print("Precision: ",metrics.precision_score(labels_test, y_pred))
print("Recall: ",metrics.recall_score(labels_test, y_pred))
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
    if y_pred[i] == - 1:
        predictions.append(0)
    else:
        predictions.append(y_pred[i])

df = pd.DataFrame({'Gene': genes})
df.insert(1,'Mutation',mutations)
df.insert(2,'Activating Status Prediction',predictions)
df.to_csv('alk_ai_results.csv',index=False,header=True)
