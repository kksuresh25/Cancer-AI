#import libraries
import numpy as np
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from svm_constructdata import constructdata
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import keras as K
import os
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # suppress CPU msg

class MyLogger(K.callbacks.Callback):
  def __init__(self, n):
    self.n = n   # print loss & acc every n epochs

  # def on_epoch_end(self, epoch, logs={}):
  #   if epoch % self.n == 0:
  #     curr_loss = logs.get('loss')
  #     curr_acc = logs.get('acc') * 100
  #     print("epoch = %4d loss = %0.6f acc = %0.2f%%" % \
  #       (epoch, curr_loss, curr_acc))

def main():
  print("Kinase Cancer Prediction Model: ")
  np.random.seed(1)

  # 1. load data into memory
  # 2. define 4-(x-x)-1 deep NN model
  # 3. compile model
  # 4. train model
  # 5. evaluate model
  # 6. make a prediction

if __name__=="__main__":
  main()

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

sm = SMOTE(random_state=seed, ratio = .75)
x_train, y_train = sm.fit_sample(features_train, labels_train)

#Create neural network
my_init = K.initializers.glorot_uniform(seed=1)
model = K.models.Sequential()
model.add(K.layers.Dense(units=8, input_dim=59, activation='tanh', kernel_initializer=my_init))
model.add(K.layers.Dense(units=8, activation='tanh', kernel_initializer=my_init))
model.add(K.layers.Dense(units=1, activation='sigmoid', kernel_initializer=my_init))

#compile model
simple_sgd = K.optimizers.SGD(lr=0.01)
model.compile(loss='binary_crossentropy', optimizer=simple_sgd, metrics=['accuracy'])

#train model (inital batch size: 32)
max_epochs = 500
my_logger = MyLogger(n=50)
h = model.fit(x_train, y_train, batch_size=32, epochs=max_epochs, verbose=0, callbacks=[my_logger])

#evaluate model on test data
np.set_printoptions(precision=4, suppress=True)
eval_results = model.evaluate(features_test, labels_test, verbose=0)
print("\nLoss, accuracy on test data: ")
print("%0.4f %0.2f%%" % (eval_results[0], \
  eval_results[1]*100))

#make predictions on test data
predictions = model.predict(features_test)
y_pred = [round(x[0]) for x in predictions]

#probability values for ROC AUC score
preds = model.predict_proba(features_test)

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
    if y_pred[i] == - 1:
        predictions.append(0)
    else:
        predictions.append(y_pred[i])

df = pd.DataFrame({'Gene': genes})
df.insert(1,'Mutation',mutations)
df.insert(2,'Activating Status Prediction',predictions)
df.to_csv('alk_ai_results.csv',index=False,header=True)
