#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 11:44:13 2019

@author: ganga
"""
# Export the necessary libraries
import pandas as pd
from pandas import DataFrame
import json

# Load the categorized comments json file
data2 = pd.read_json("controversial-comments.jsonl", lines = True)

# Export regular expression library and natural language tool kit
import re
import nltk
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

data3 = data2.head()

ps = PorterStemmer()
# Function to extract only the alphabetical characters
def commas(txt):
    txt = re.sub('[^a-zA-Z]', " ", txt)
    return txt

# Applying the function on the condensed data frame
data2["txt"] = data2["txt"].apply(commas)

# Passing on the processed data on to the count vectorizer
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words = {'english'}, max_features = 1000)
X = cv.fit_transform(data2["txt"]).toarray()
y = data2.iloc[:, 0].values

# Splitting the data into test and train data
from sklearn.cross_validation import train_test_split


# Calculating the test and training split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 1)


from sklearn.linear_model import LogisticRegression

# Applying logistic regression with l2 penalty
classifier = LogisticRegression(random_state = 0, penalty = "l2")
classifier.fit(X_train, y_train)

# Calculating the predictions for the test set
y_pred = classifier.predict(X_test)

# Calculating the accuracy of the model
from sklearn.metrics import accuracy_score

# Calculating the test and train predictions
test_predictions = classifier.predict(X_test)
train_predictions = classifier.predict(X_train)


# Calculating the test and training accuracies
accuarcy_test = accuracy_score(y_test, test_predictions)
accuarcy_train = accuracy_score(y_train, train_predictions)

# Calculating the area under the curve
from sklearn import metrics
auc = metrics.roc_auc_score(y_test, y_pred)


# Calculating the classification report
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)


# Calculating the accuracies from 3 fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train,  y = y_train, n_jobs = -1, cv = 3)
