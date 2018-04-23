# Based on ideas from Nick Becker at https://github.com/beckernick/logistic_regression_from_scratch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


column_names = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
            'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

# Importing training data
df_train = pd.read_csv("file://localhost/home/matthew/Downloads/train.csv", names=column_names, skiprows=1, nrows=500)
# Dropping features that are unlikely to have an effect on survival
df_train = df_train.drop(['Name','Ticket','PassengerId','Cabin'], axis=1)
# Get rid of all rows with missing data, this is suboptimal, but useful for initial testing
df_train = df_train.dropna(how="any", axis=0)
# Changing categorical features into one-hot form
eS = pd.Series(np.zeros(len(df_train), dtype=int),index=df_train.index)
df_train = df_train.assign(eS=eS.values)
df_train = df_train.assign(eC=eS.values)
df_train = df_train.assign(eQ=eS.values)
df_train = df_train.assign(female=eS.values)

df_train.loc[df_train.Sex=='female', 'female'] = 1
df_train.loc[df_train.Embarked=='S', 'eS'] = 1
df_train.loc[df_train.Embarked=='C', 'eC'] = 1
df_train.loc[df_train.Embarked=='Q', 'eQ'] = 1
# Removed the redundant categorical features
df_train = df_train.drop(['Sex','Embarked'],axis=1)
# Separating the dependent variable/label from the predictors
df_features = df_train.drop(['Survived'],axis=1)
df_label = df_train['Survived']

# The code expects numpy arrays as inputs
features_array = np.asarray(df_features)
label_array = np.asarray(df_label, dtype=int)

np.random.seed(12)
num_observations = len(label_array)

features = features_array

labels = label_array

#print(labels)

def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))

def logistic_regression(features, target, num_steps, learning_rate, add_intercept = False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))

    weights = np.zeros(features.shape[1])

    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with log likelihood gradient
        output_error_signal = target - predictions

        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient

    return weights

weights = logistic_regression(features, labels,
                     num_steps = 50000, learning_rate = 5e-5, add_intercept=True)

print(weights)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(fit_intercept=True, C = 1e15)
clf.fit(features, labels)

print(clf.intercept_, clf.coef_)
print(weights)
final_scores = np.dot(np.hstack((np.ones((features.shape[0], 1)),
                                 features)), weights)


preds = np.round(sigmoid(final_scores))
# print(preds)
print('Accuracy from scratch: {0}'.format((preds == labels).sum().astype(float) / len(preds)))
print('Accuracy from sk-learn: {0}'.format(clf.score(features, labels)))
