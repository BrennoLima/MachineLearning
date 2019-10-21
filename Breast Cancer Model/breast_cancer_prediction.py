# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 11:34:55 2019

@author: 809438
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
Dataset = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))

# Data Housekeeping
Dataset.describe()
Dataset.isna().any()
Dataset.head()
Dataset.columns

# Histogram
Dataset2 = Dataset.drop(columns = ["target"])
Dataset2.corrwith(Dataset.target).plot.bar(title = "Correlation with target", 
                 grid = True)

# X and y
X = Dataset2
y = Dataset["target"]

# Splitting X_train, X_test, y_train, y_test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train2 = pd.DataFrame(sc_x.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_x.transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2
    
# SVM - Support Vector Machine - Linear
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Score Dataframe SVC - Linear
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

results = pd.DataFrame([["SVC - Linear", acc, prec, rec, f1]], 
                       columns = ["Model", "Accuracy", "Precision", "Recall", "F1"])

# SVM - Support Vector Machine - Rbf
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Score Dataframe SVC - Rbf
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([["SVC - RBF", acc, prec, rec, f1]], 
                       columns = ["Model", "Accuracy", "Precision", "Recall", "F1"])
results = results.append(model_results, ignore_index = True)

# Random Forest - Gini
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'gini')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Score Dataframe Random Forest - Gini
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([["Random Forest - Gini", acc, prec, rec, f1]], 
                       columns = ["Model", "Accuracy", "Precision", "Recall", "F1"])
results = results.append(model_results, ignore_index = True)

# Random Forest - Entropy
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Score Dataframe Random Forest - Entropy
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([["Random Forest - Entropy", acc, prec, rec, f1]], 
                       columns = ["Model", "Accuracy", "Precision", "Recall", "F1"])
results = results.append(model_results, ignore_index = True)

# Best model = SVC - RBF
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# K-fold Cross Validation - SVC - RBF
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("acc = %0.2f +/- %0.2f" %(accuracies.mean(), accuracies.std()*2))
# Accuracie is 95% - 100%

# Grid Search SVC - RBF - Round 1
parameters = {"C" : [0.5, 1, 1.5],
              "gamma": ['auto', 1, 0.1, 0.01, 0.001]}
from sklearn.model_selection import GridSearchCV
grid_search= GridSearchCV(estimator = classifier, param_grid = parameters, 
                          scoring = "accuracy", cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
grid_search.best_score_
grid_search.best_params_

# Grid Search SVC - RBF - Round 2
parameters = {"C" : [1.5, 2.5, 3.5],
              "gamma": ['auto']}
from sklearn.model_selection import GridSearchCV
grid_search= GridSearchCV(estimator = classifier, param_grid = parameters, 
                          scoring = "accuracy", cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
grid_search.best_score_
grid_search.best_params_

# Prediction with Grid Search
y_pred = grid_search.predict(X_test)

# Score Dataframe Grid Search RBF 2 rounds
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([["RBF - GridSearch 2x", acc, prec, rec, f1]], 
                       columns = ["Model", "Accuracy", "Precision", "Recall", "F1"])
results = results.append(model_results, ignore_index = True)

# K-fold Cross Validation - SVC RBF - GridSearch 2 rounds
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("acc = %0.2f +/- %0.2f" %(accuracies.mean(), accuracies.std()*2))
# Accuracie is 95% - 100%, Parameter tunning made no difference in this model

# Visualization
cm = confusion_matrix(y_test, y_pred)
sn.heatmap(cm, annot = True)

# Prediction for the whole Dataset
# Dataset scalling
sc_x = StandardScaler()
X2 = pd.DataFrame(sc_x.fit_transform(X))
X2.columns = X.columns.values
X2.index = X.index.values
X = X2

# Prediction with Grid Search
y_pred = grid_search.predict(X)

# Visualization
cm = confusion_matrix(y, y_pred)
sn.heatmap(cm, annot = True)

# Score Dataframe Grid Search RBF 2 rounds - Whole dataset
acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

model_results = pd.DataFrame([["RBF - GridSearch 2x - Full", acc, prec, rec, f1]], 
                       columns = ["Model", "Accuracy", "Precision", "Recall", "F1"])
results = results.append(model_results, ignore_index = True)







    
    
    
    
    
    
    
    