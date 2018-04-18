import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


column_names = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
            'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

df_train = pd.read_csv("file://localhost/home/matthew/Downloads/train.csv", names=column_names, skiprows=1, nrows=500)

df_train = df_train.drop(['Name','Ticket','PassengerId','Cabin'], axis=1)

# df_train = df_train.dropna(how="any", axis=0)
nanSum = df_train.isnull().sum()
print(nanSum)
ages = df_train['Age']
print(ages)
ages = ages.fillna(-1)
fig, ax = plt.subplots()
ax.hist(ages, bins=20)
plt.show()
# print(df_train)

eS = pd.Series(np.zeros(len(df_train), dtype=int),index=df_train.index)
# eC = pd.Series(np.zeros(len(df_train), dtype=int),index=df_train.index)
# eQ = pd.Series(np.zeros(len(df_train), dtype=int),index=df_train.index)
# male = pd.Series(np.zeros(len(df_train), dtype=int),index=df_train.index)
# female = pd.Series(np.zeros(len(df_train), dtype=int),index=df_train.index)

df_train = df_train.assign(eS=eS.values)
df_train = df_train.assign(eC=eS.values)
df_train = df_train.assign(eQ=eS.values)
df_train = df_train.assign(male=eS.values)
df_train = df_train.assign(female=eS.values)

# print(df_train)
df_train.loc[df_train.Sex=='male', 'male'] = 1
df_train.loc[df_train.Sex=='female', 'female'] = 1
df_train.loc[df_train.Embarked=='S', 'eS'] = 1
df_train.loc[df_train.Embarked=='C', 'eC'] = 1
df_train.loc[df_train.Embarked=='Q', 'eQ'] = 1
# print(df_train)
df_train = df_train.drop(['Sex','Embarked'],axis=1)
# print(df_train)
df_features = df_train.drop(['Survived'],axis=1)
df_label = df_train['Survived']
# print(df_features)
# print(df_label)
features_array = np.asarray(df_features)
# print(features_array)
