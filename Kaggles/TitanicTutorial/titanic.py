# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 20:01:21 2017

@author: Matthew
"""
import tempfile
import tensorflow as tf
import numpy as np
import pandas as pd

#Kaggle Titanic intro competition

data_file = "file://localhost/Downloads/train.csv"
df_train = pd.read_csv("file://localhost/Downloads/train.csv")
df_test = pd.read_csv("file://localhost/Downloads/test.csv")


#feature columns
gender = tf.feature_column.categorical_column_with_vocabulary_list(
    "gender", ["female", "male"])
embarked = tf.feature_column.categorical_column_with_vocabulary_list(
    "Embarked", ["Q", "S", "C"])
age = tf.feature_column.numeric_column("Age")
pclass = tf.feature_column.numeric_column("Pclass")
fare = tf.feature_column.numeric_column("Fare")
sibsp = tf.feature_column.numeric_column("SibSp")
parch = tf.feature_column.numeric_column("Parch")

base_columns = [gender, embarked, age, pclass, fare, sibsp, parch]
crossed_columns = [tf.feature_column.crossed_column(
        ["gender", "pclass"], hash_bucket_size=1000)]

def input_fn(data_file, num_epochs, shuffle):
    df_train = pd.read_csv(data_file)
    df_train = df_train.drop(['Name','Ticket','PassengerId','Cabin'], axis=1)
    df_train = df_train.dropna(how="any", axis=0)
    labels = df_train.filter(items=['Survived'])
    df_train = df_train.drop(['Survived'], axis=1)
    return tf.estimator.inputs.pandas_input_fn(
      x=df_train,
      y=labels,
      batch_size=100,
      num_epochs=num_epochs,
      shuffle=shuffle,
      num_threads=5)
    
model_dir = tempfile.mkdtemp()
m = tf.estimator.LinearClassifier(
    model_dir=model_dir, feature_columns=base_columns + crossed_columns)