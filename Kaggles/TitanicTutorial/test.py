import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model,svm
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, auc
from sklearn.preprocessing import PolynomialFeatures

column_names = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
            'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

df = pd.read_csv("file://localhost/home/matthew/Downloads/train.csv", names=column_names, skiprows=1)

df = df.drop(['Name','Ticket','PassengerId','Cabin'], axis=1)

# df = df.dropna(how="any", axis=0)
# nanSum = df.isnull().sum()
# print(nanSum)
# ages = df['Age']
# ages = ages.fillna(-1)
# fig, ax = plt.subplots()
# ax.hist(ages, bins=20)
# plt.show()
# print(df)

eS = pd.Series(np.zeros(len(df), dtype=int),index=df.index)

df = df.assign(eS=eS.values)
df = df.assign(eC=eS.values)
df = df.assign(eQ=eS.values)
df = df.assign(female=eS.values)
df = df.assign(missingAge=eS.values)
df = df.assign(Parch0=eS.values)
df = df.assign(Parch1=eS.values)
df = df.assign(Parch2=eS.values)
df = df.assign(Parch3=eS.values)
df = df.assign(Parch4=eS.values)
df = df.assign(Parch5=eS.values)


df.loc[df.Sex=='female', 'female'] = 1
df.loc[df.Embarked=='S', 'eS'] = 1
df.loc[df.Embarked=='C', 'eC'] = 1
df.loc[df.Embarked=='Q', 'eQ'] = 1
df.loc[df.Parch==0, 'Parch0'] = 1
df.loc[df.Parch==1, 'Parch1'] = 1
df.loc[df.Parch==2, 'Parch2'] = 1
df.loc[df.Parch==3, 'Parch3'] = 1
df.loc[df.Parch==4, 'Parch4'] = 1
df.loc[df.Parch==5, 'Parch5'] = 1

# create a feature for missing Age data
df.loc[df.Age.isnull(), 'missingAge'] = 1

df = df.drop(['Sex','Embarked'],axis=1)

# split data on whether Age is missing or not

df_age_n = df.loc[df.Age.isnull()]
df_age_y = df.loc[df.Age >= 0]

# fig, ax = plt.subplots(2, 4, sharey=True)
#
# df_age_y.boxplot(column='Age', by='Parch', ax=ax[0,0])
# df_age_y.boxplot(column='Age', by='SibSp', ax=ax[0,1])
# df_age_y.boxplot(column='Age', by='Pclass', ax=ax[0,2])
# df_age_y.boxplot(column='Age', by='female', ax=ax[0,3])
# df_age_y.boxplot(column='Age', by='eS', ax=ax[1,0])
# df_age_y.boxplot(column='Age', by='eC', ax=ax[1,1])
# df_age_y.boxplot(column='Age', by='eQ', ax=ax[1,2])
# df_age_y.boxplot(column='Age', by='Survived', ax=ax[1,3])
#
# plt.tight_layout()
# plt.show()

ages = np.asarray(df_age_y.Age)
nTrainAges = int(len(ages)*0.7)


ages_train = ages[:nTrainAges]
ages_test = ages[nTrainAges:]
print("Size of training set for Age regression: " + str(nTrainAges))
age_predictors_train = df_age_y.drop(
        ['Age'],axis=1).drop(
        ['Parch'],axis=1).drop(
        ['Survived'],axis=1).drop(
        ['eS'],axis=1).drop(
        ['eC'],axis=1).drop(
        ['eQ'],axis=1).drop(
        ['Fare'],axis=1).drop(
        # ['female'],axis=1).drop(
        ['missingAge'],axis=1)[:nTrainAges]
age_predictors_test = df_age_y.drop(
        ['Age'],axis=1).drop(
        ['Parch'],axis=1).drop(
        ['Survived'],axis=1).drop(
        ['eS'],axis=1).drop(
        ['eC'],axis=1).drop(
        ['eQ'],axis=1).drop(
        ['Fare'],axis=1).drop(
        # ['female'],axis=1).drop(
        ['missingAge'],axis=1)[nTrainAges:]


# print(age_predictors_train.head())
regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(age_predictors_train,ages_train)

predicted_ages = regr.predict(age_predictors_test)
print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f"
      % mean_squared_error(ages_test, predicted_ages))
print('Variance score: %.2f' % r2_score(ages_test, predicted_ages))

# fig2, ax2 = plt.subplots(1,2, sharey=True, sharex=True)
# ax2[0].hist(ages_test, range=(0,90))
# ax2[1].hist(predicted_ages, range=(0,90))
# plt.show()
predictor_list = ['Pclass',  'SibSp',  'female',  'Parch0',  'Parch1',  'Parch2',  'Parch3',  'Parch4',  'Parch5']

def fillPredictedAge(model, predictor_list, df, index):
    droplist = []
    for column in list(df):
        if column not in predictor_list:
            droplist.append(column)
    tmp_df = df.drop(droplist, axis=1)
    new_age = model.predict(tmp_df.loc[index].values.reshape(1,-1))

    return new_age


for row in df.itertuples():
    if row[11]==1:
        index = row[0]
        # print(index)
        new_age = fillPredictedAge(regr, predictor_list, df, index)
        # print(new_age)
        df.loc[index, 'Age'] = new_age

df_features = df.drop(['Survived'],axis=1)
df_labels = df['Survived']

nTrain70 = int(len(df)*0.7)

df_features_train = df_features[:nTrain70]
df_labels_train = df_labels[:nTrain70]
df_features_test = df_features[nTrain70:]
df_labels_test = df_labels[nTrain70:]

LR = linear_model.LogisticRegression(fit_intercept=True, C = 1e15)

features_train = np.asarray(df_features_train)
labels_train = np.asarray(df_labels_train)
features_test = np.asarray(df_features_test)
labels_test = np.asarray(df_labels_test)

LR.fit(features_train, labels_train)
# print(LR.intercept_, LR.coef_)
print('Accuracy from sk-learn LR: {0}'.format(LR.score(features_train, labels_train)))

# C = 1.0 # SVM regularization parameter
#
# SVM = svm.SVC(kernel='linear', C=C,gamma='auto')
# SVM.fit(features_train, labels_train)
# print('Accuracy from sk-learn SVM: {0}'.format(SVM.score(features_train, labels_train)))

poly = PolynomialFeatures(2)

features_train_poly = poly.fit_transform(features_train)
features_test_poly = poly.fit_transform(features_test)

LRpoly = linear_model.LogisticRegression(fit_intercept=True, C = 1e15)
LRpoly.fit(features_train_poly, labels_train)

scores = LRpoly.fit(features_train_poly, labels_train).predict_proba(features_test_poly)

print('Accuracy from sk-learn LR with polynomial features: {0}'.format(LRpoly.score(features_train_poly, labels_train)))

# print(labels_test)
# print(scores[:,1])

fpr, tpr, thresholds = roc_curve(labels_test, scores[:, 1])
roc_auc = auc(fpr, tpr)
print(roc_auc)
# print(tpr)
# print(fpr)
# print(thresholds)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
