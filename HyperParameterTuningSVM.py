#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 20:48:19 2021

@author: ayeshauzair
"""

# Apply hyperparameter tuning (3 methods - grid, random and bayesian search) on SVM model
# for the heart dataset and compare the best params and best score.


import pandas as pd
import pandas_profiling as pp
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn.svm import SVC, SVR
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from skopt.space import Real, Categorical, Integer


# Import Dataset
df = pd.read_csv("heart.csv")
print(df.head())
print(df.info())
print(df.describe())


# Pandas Profiling
# profile = pp.ProfileReport(df)
# profile.to_file("heart_EdA.html")


# Mean values to replace 0s
mean_Cholesterol = df["Cholesterol"].mean()
mean_RestingBP = df["RestingBP"].mean()
df["Cholesterol"] = df["Cholesterol"].replace({0:mean_Cholesterol})
df["RestingBP"] = df["RestingBP"].replace({0:mean_RestingBP})


# Convert boolean objects to numerical values
df["ExerciseAngina"] = df["ExerciseAngina"].replace({"Y":"1"})
df["ExerciseAngina"] = df["ExerciseAngina"].replace({"N":"0"})
df["ExerciseAngina"] = df["ExerciseAngina"].astype(int)
df["Sex"] = df["Sex"].replace({"M":"0"})
df["Sex"] = df["Sex"].replace({"F":"1"})
df["Sex"] = df["Sex"].astype(int)


# Create dummies for categorical columns 
df = pd.get_dummies(df,columns=["ChestPainType","RestingECG","ST_Slope"])
print("\n\n--------------------------------------------------------------------------")
print("After Data Conversion: ")
print(df.info())


# Prepare Data for Modeling
y = df["HeartDisease"]
X = df.drop("HeartDisease",axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=0)


# Support Vector Machine modelling before hyperparameter tuning
# Available kernels {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'

# Simple Linear Model Evaluation
model_SVC = SVC(kernel="linear")
model_SVC.fit(X_train,y_train)
y_pred_SVC = model_SVC.predict(X_test)
cnfn_SVC = confusion_matrix(y_test,y_pred_SVC)
cr_SVC = classification_report(y_test,y_pred_SVC)
acc_SVC = accuracy_score(y_test, y_pred_SVC)

print("\n\n--------------------------------------------------------------------------")
print("Support Vector Machine algorithm (Linear)")
print(cnfn_SVC)
print(cr_SVC)
print("Accuracy of SVM is: ",acc_SVC)
print("\n\n--------------------------------------------------------------------------")


# General Search Space
param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': ['scale'],
              'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
              'degree': [1,2,3,4,5,6]}


# Linear Search Space
param_grid_linear = {'C': [0.1, 1, 10, 100], 
              'gamma': ["scale"],
              'kernel': ['linear']}


# Non-Linear Search Space
param_grid_non_linear = {'C': [0.1, 1, 10], 
              'gamma': ["auto", "scale"],
              'kernel': ['rbf', 'poly', 'sigmoid'],
              'degree': [1,2,3,4,5,6]}


# Random Search CV
tuning_random = RandomizedSearchCV(SVC(), 
                                    param_grid,
                                    scoring="accuracy", 
                                    cv=5,
                                    n_jobs=-1, 
                                    refit = True,
                                    verbose=2,
                                    random_state=0) 
tuning_random.fit(X_train,y_train)
print("\n\n--------------------------------------------------------------------------")
print("Random Search")
print(tuning_random.best_params_)
print(tuning_random.best_score_)
print(tuning_random.best_estimator_)
print("\n\n--------------------------------------------------------------------------")
    

# Grid Search CV
tuning_grid = GridSearchCV(SVC(), 
                          param_grid,
                          scoring="accuracy",
                          n_jobs=-1,
                          verbose=2)
tuning_grid.fit(X_train,y_train)
print("\n\n--------------------------------------------------------------------------")
print("Grid Search")
print(tuning_grid.best_params_)
print(tuning_grid.best_score_)
print(tuning_grid.best_estimator_)
print("\n\n--------------------------------------------------------------------------")


# Bayes Search CV
tuning_bayes = BayesSearchCV(SVC(),
                          param_grid,
                          scoring="accuracy",
                          cv=5,
                          n_jobs=-1,
                          verbose=2)
tuning_bayes.fit(X_train,y_train)
print("\n\n--------------------------------------------------------------------------")
print("Bayes Search")
print(tuning_bayes.best_params_)
print(tuning_bayes.best_score_)
print(tuning_bayes.best_estimator_)
print("\n\n--------------------------------------------------------------------------")


