# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 20:01:21 2017

@author: Matthew
"""
import tempfile
#import argparse
import sys
import tensorflow as tf
import numpy as np
import pandas as pd

#Kaggle Titanic intro competition

#Need to create train and test datasets out of the train file
column_names = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
            'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

df_train = pd.read_csv("file://localhost/home/matthew/Downloads/train.csv", names=column_names, skiprows=1, nrows=500)
df_test = pd.read_csv("file://localhost/home/matthew/Downloads/train.csv", names=column_names, skiprows=501)


#feature columns
gender = tf.feature_column.categorical_column_with_vocabulary_list(
    "Sex", ["female", "male"])
embarked = tf.feature_column.categorical_column_with_vocabulary_list(
    "Embarked", ["Q", "S", "C"])
age = tf.feature_column.numeric_column("Age")
pclass = tf.feature_column.numeric_column("Pclass")
fare = tf.feature_column.numeric_column("Fare")
sibsp = tf.feature_column.numeric_column("SibSp")
parch = tf.feature_column.numeric_column("Parch")

base_columns = [gender, age, embarked, pclass, fare, sibsp, parch]
crossed_columns = [tf.feature_column.crossed_column(["Sex", "Pclass"], hash_bucket_size=1000)]

def build_estimator(model_dir):
    m = tf.estimator.LinearClassifier(
      model_dir=model_dir,
      feature_columns=base_columns + crossed_columns)
#      feature_columns=base_columns)

    return m


def input_fn(df_data, num_epochs, shuffle):
    # df_data = pd.read_csv(data_file)
    df_data = df_data.drop(['Name','Ticket','PassengerId','Cabin'], axis=1)
    df_data = df_data.dropna(how="any", axis=0)

    labels = df_data.filter(items=['Survived'])
    #df_data = df_data.drop(['Survived'], axis=1)
    return tf.estimator.inputs.pandas_input_fn(x=df_data, y=labels,
        batch_size=100, num_epochs=num_epochs, shuffle=shuffle,
        num_threads=1)

def train_and_eval(model_dir, train_steps, train_data, test_data):
    #model_dir = tempfile.mkdtemp() if not model_dir else model_dir
    m = build_estimator(model_dir)

    m.train(input_fn=input_fn(train_data, num_epochs=None, shuffle=True),
      steps=train_steps)
    results = m.evaluate(input_fn(test_data, num_epochs=1, shuffle=False),
      steps=None)
    print("model directory = %s" % model_dir)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))

train_and_eval('./tmp', 2000, df_train, df_test)
