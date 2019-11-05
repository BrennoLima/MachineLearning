# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:46:36 2019

@author: 809438
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

# Importing data
training_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')
test_id = test_set.PassengerId

# Visualization and cleaning 
sn.heatmap(training_set.isnull(), yticklabels = False, cbar = False, cmap = 'Blues')
training_set.Name.head(5)




training_set = training_set.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Cabin'])

sn.heatmap(test_set.isnull(), yticklabels = False, cbar = False, cmap = 'Blues')
test_set = test_set.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Cabin'])

test_set.isnull().any()
test_set[test_set['Fare'].isnull()]
test_set['Fare'] = test_set['Fare'].fillna(test_set['Fare'].median())

training_set.isnull().any()
training_set[training_set['Embarked'].isnull()]
sn.countplot(x = 'Embarked', data = training_set )
training_set['Embarked'] = training_set['Embarked'].fillna('S')

# Age distribution before model
training_set['Age'].hist(bins = 20)

training_set[training_set['Age'].isnull()] # 177 values are null
test_set[test_set['Age'].isnull()]         # 86 values are null 

# Machine Learning model to predict ages 
dataset = training_set
dataset = dataset.drop(columns = ['Survived'])
dataset = dataset.append(test_set)
sn.countplot(x = 'Age', data = dataset)
dataset[dataset['Age'].isnull()]        # 263 values are null

X = dataset[dataset['Age'].notnull()]
X_test = dataset[dataset['Age'].isnull()] # this is the values to be filled
X_test_Age = X_test.Age
X_test = X_test.drop(columns = ['Age'])

X_train = X.drop(columns = ['Age'])
y_train = X.Age

# Categorical values
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
X_train = X_train.drop(columns = ['Sex_female', 'Embarked_Q'])
X_test = X_test.drop(columns = ['Sex_female', 'Embarked_Q'])

# scalling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train2 = pd.DataFrame(sc_x.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_x.transform(X_test))
X_train2.columns = X_train.columns.values
X_train2.index = X_train.index.values
X_test2.columns = X_test.columns.values
X_test2.index = X_test.index.values
X_test = X_test2
X_train = X_train2

# model SVR
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)
years_prediction = pd.DataFrame(regressor.predict(X_test))

# Model Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100)
regressor.fit(X_train, y_train)
years_prediction2 = pd.DataFrame(regressor.predict(X_test))

# Visualization of the year prediction, the second looks more distributed 
years_prediction2.hist(bins = 20)
years_prediction.hist(bins = 20)

years_prediction.index = X_test_Age.index.values
# index 0-176 are from training set
# index 177-262 are from test set
years_prediction_training = years_prediction.iloc[0:177, :]
years_prediction_test = years_prediction.iloc[177:263, :]

ypdtrain = years_prediction_training.iloc[:, :].values
ypdtest = years_prediction_test.iloc[:, :].values
# Join the new predictions to the training_set and test_set

np.isnan(X.iloc[[0],[0]]).values.item()

j = 0
for i in range(0, 418):
    if np.isnan(test_set.iloc[[i],[2]]).values.item():
        test_set.iloc[[i],[2]] = ypdtest[j].item()
        j += 1
        
j = 0
for i in range(0, 891):
    if np.isnan(training_set.iloc[[i],[3]]).values.item():
        training_set.iloc[[i],[3]] = ypdtrain[j].item()
        j += 1

# Visualization after cleaning
training_set['Age'].hist(bins = 20)
test_set['Age'].hist(bins = 20)

# Categorical data
training_set = pd.get_dummies(training_set)
test_set = pd.get_dummies(test_set)

training_set = training_set.drop(columns = ['Sex_male', 'Embarked_C'])
test_set = test_set.drop(columns = ['Sex_male', 'Embarked_C'])

# Splitting
X_train = training_set.drop(columns = ['Survived'])
y_train = training_set.Survived
X_test = test_set


# Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train2 = pd.DataFrame(sc_x.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_x.transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_test2.index = X_test.index.values
X_train2.index = X_train.index.values
X_train = X_train2
X_test = X_test2

# Random Forest Classifier 84% +/- 10%
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy',
                                    max_features = 5, max_depth = 10, min_samples_split = 4,
                                    min_samples_leaf = 2, bootstrap = True)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# k-fold 
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, scoring = "accuracy",
                             cv = 10)
print("acc = %0.2f +/- %0.2f" % (accuracies.mean(), accuracies.std()*2))


# Grid Search - parameters found after 4 rounds
parameters = {"max_features": [5],
              "max_depth": [10],
              "min_samples_split" : [4],
              "min_samples_leaf" : [2],
              "bootstrap": [True],
              "criterion" : ['entropy']
              }

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = "accuracy", cv = 10)
grid_search = grid_search.fit(X_train, y_train)
grid_search.best_params_

test_id = test_id.to_frame()
output = pd.DataFrame({"PassengerId": test_id.PassengerId, "Survived": y_pred})
output.PassengerId = output.PassengerId.astype(int)
output.Survived = output.Survived.astype(int)

output.to_csv("output.csv", index = False)











    
        
