import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import pandas as pd


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
df_train.loc[df_train.Sex=='male', 'male'] = 1
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

X = features_array
y = label_array

C = 1.0 # SVM regularization parameter
model = svm.SVC(kernel='linear', C=C,gamma='auto')
model.fit(X,y)
