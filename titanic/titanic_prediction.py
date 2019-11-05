# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:14:15 2019

@author: 809438
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')

dataset_train.isna().any() # Age, cabin, embarked
dataset_test.isna().any() # Age, Fare, Cabin

dataset_train['Embarked'] = dataset_train['Embarked'].fillna('S') # train - fix embarked
dataset_train['Age'] = dataset_train['Age'].fillna(dataset_train['Age'].median()) # train - fix age
dataset_train = dataset_train.drop(columns = ['Cabin']) # train - fix Cabin


dataset_test['Fare'] = dataset_test['Fare'].fillna(dataset_test['Fare'].median()) # test - fix Fare missing
dataset_test['Age'] = dataset_test['Age'].fillna(dataset_test['Age'].median()) # test - fix Age missing
dataset_test = dataset_test.drop(columns = ['Cabin']) # test - fix Cabin

# id/name
train_id = dataset_train.PassengerId
train_name = dataset_train.Name
dataset_train = dataset_train.drop(columns = ['PassengerId', 'Name', 'Ticket'])
y_train = dataset_train.Survived
dataset_train = dataset_train.drop(columns = ['Survived'])
X_train = dataset_train

test_id = dataset_test.PassengerId
test_name = dataset_test.Name
dataset_test = dataset_test.drop(columns = ['PassengerId', 'Name', 'Ticket'])
X_test = dataset_test


# Categorical variables
X_train = pd.get_dummies(X_train)
X_train = X_train.drop(columns = ['Sex_male', 'Embarked_Q'])
X_test = pd.get_dummies(X_test)
X_test = X_test.drop(columns = ['Sex_male', 'Embarked_Q'])

# Feature scalling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train2 = pd.DataFrame(sc_x.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_x.transform(X_test))
X_train2.columns = X_train.columns.values
X_train2.index = X_train.index.values
X_test2.columns = X_test.columns.values
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2

# SVM - rbf k-fold = 82% +/- 7%
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
y_pred_SVM = classifier.predict(X_test)
y_pred_SVM.sum()

# Random Forest  k-fold = 81% +/- 9%
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
classifier.fit(X_train, y_train)
y_pred_RF = classifier.predict(X_test)
y_pred_RF.sum()

# Logistic Regression = 80% +/- 9%
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred_LR = classifier.predict(X_test)
y_pred_LR.sum()

# k-fold 
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, scoring = "accuracy",
                             cv = 10, n_jobs = - 1)
print("acc = %0.2f +/- %0.2f" % (accuracies.mean(), accuracies.std()*2))

# Grid Search
parameters = {"penalty": ["l1"],
              "C": [3],
              "fit_intercept" : [True]
              }
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = "accuracy", 
                  cv = 10, n_jobs = -1)
gs = gs.fit(X_train, y_train)
gs.best_score_
gs.best_params_

y_pred = gs.predict(X_test)
test_id = test_id.to_frame()
output = pd.DataFrame({"PassengerId": test_id.PassengerId, "Survived": y_pred})
output.PassengerId = output.PassengerId.astype(int)
output.Survived = output.Survived.astype(int)

output.to_csv("output.csv", index = False)









