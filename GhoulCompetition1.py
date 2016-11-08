# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 13:41:46 2016

@author: Zachery McKinnon
"""
import pandas as pd
import csv
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier

test_df = pd.read_csv("...\Testing_sample.csv")
train_df = pd.read_csv("...\Training_sample.csv")

y_train = train_df.ix[:,6].values
x1_train = train_df.ix[:,0:5].values
for elem in train_df['color'].unique():
    train_df[str(elem)] = train_df['color'] == elem
x2_train = train_df.ix[:,7:].values
X_train = np.concatenate((x1_train, x2_train), axis=1)

x1_test = test_df.ix[:,0:5].values
for elem in test_df['color'].unique():
    test_df[str(elem)] = test_df['color'] == elem
x2_test = test_df.ix[:,6:].values
X_test = np.concatenate((x1_test, x2_test), axis=1)

# Split the training data into two sets for testing
X_train1 = X_train[:-40]
X_train2 = X_train[-40:]

# Split the targets into two sets for testing
y_train1 = y_train[:-40]
y_train2 = y_train[-40:]

# Choose classifiers (and ensembles), in this case random forest and gaussian Naive Bayes
# were chosen as base estimators. They were compared to a few other classifiers, which were less effective

param_grid = {"base_estimator__max_depth": [3, 10],
              "base_estimator__max_features": [1, 3, 10],
              "base_estimator__min_samples_split": [1, 3, 10],
              "base_estimator__min_samples_leaf": [1, 3, 10],
              "base_estimator__bootstrap": [True, False],
              "base_estimator__criterion": ["gini", "entropy"],}

clf1 = RandomForestClassifier(n_estimators=20)
clf2 = GaussianNB()
eclf1 = VotingClassifier(estimators=[('rf', clf1), ('nb', clf2)], voting='hard')
eclf2 = BaggingClassifier(AdaBoostClassifier(clf1, n_estimators=50, algorithm='SAMME.R'))
eclf3 = AdaBoostClassifier(clf1, n_estimators = 50, algorithm='SAMME.R')
eclf4 = GridSearchCV(eclf3, param_grid=param_grid)

# Examine the score of each classifier
print clf1.fit(X_train1, y_train1).score(X_train2, y_train2)
print clf2.fit(X_train1, y_train1).score(X_train2, y_train2)
print eclf1.fit(X_train1, y_train1).score(X_train2, y_train2)
print eclf2.fit(X_train1, y_train1).score(X_train2, y_train2)
print eclf3.fit(X_train1, y_train1).score(X_train2, y_train2)
print eclf4.fit(X_train1, y_train1).score(X_train2, y_train2)

# Refit eclf4, the best classifier, on all of the training data
eclf4.fit(X_train, y_train)

# Convert the classifier to csv
y_test_eclf4 = eclf4.predict(X_test)
id_test = test_df.ix[:,0].values
result = zip(id_test, y_test_eclf4.T)
with open('results_eclf4.csv','wb') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['id','type'])
    for row in result:
        csv_out.writerow(row)

