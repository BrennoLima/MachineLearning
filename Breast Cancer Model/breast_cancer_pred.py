# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 13:56:41 2019

@author: 809438
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))
        

X = df_cancer.drop(['target'], axis = 'columns')
y = df_cancer['target']

# split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scalling
min_train = X_train.min()
range_train = (X_train - min_train).max()
X_train_scalled = (X_train - min_train)/range_train

X_test_scalled = (X_test - min_train)/range_train

# Model SVC support vector machine
from sklearn.svm import SVC
classifier = SVC(C = 7, gamma = 0.3)
classifier.fit(X_train_scalled, y_train)

# model Random Forest 
from sklearn.ensemble import RandomForestClassifier
forest_classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
forest_classifier.fit(X_train_scalled, y_train)

# Predict
y_pred = classifier.predict(X_test_scalled)
y_pred_forest = forest_classifier.predict(X_test_scalled)

# Analysis 
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.model_selection import cross_val_score

# SVC
cm = confusion_matrix(y_test, y_pred)

accuracies = cross_val_score(classifier, X_train_scalled, y_train, cv = 10)
print(classification_report(y_test, y_pred))
print("Mean accuracy SVC = ", accuracies.mean())

# Random forest
cm_forest = confusion_matrix(y_test, y_pred_forest)
accuracies_forest = cross_val_score(forest_classifier, X_train_scalled, y_train, cv = 10)
print("Mean accuracy Random forest = ", accuracies_forest.mean())


# Parameter tunning  Grid Search
from sklearn.model_selection import GridSearchCV
parameters = [{ 'C': [1,6,7,8], 'gamma':[0.1, 0.2, 0.3] }]
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)
grid_search = grid_search.fit(X_train_scalled, y_train)
best_params = grid_search.best_params_
print("Best params for SVC = ", best_params)











