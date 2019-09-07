#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 11:51:30 2019

@author: ganga
"""
# Export the necessary libraries
import pandas as pd
from pandas import DataFrame
import json

# Load the categorized comments json file
data3 = pd.read_json("controversial-comments.jsonl", lines = True)

# Export regular expression library and natural language tool kit
import re
import nltk
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

data4 = data3.head()

ps = PorterStemmer()

# Function to extract only the alphabetical characters
def commas(txt):
    txt = re.sub('[^a-zA-Z]', " ", txt)
    return txt
# Applying the function on the condensed data frame
data3["txt"] = data3["txt"].apply(commas)

# Passing on the processed data on to the count vectorizer
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words = {'english'}, max_features = 1000)
X = cv.fit_transform(data3["txt"]).toarray()
y = data3.iloc[:, 0].values

# Splitting the data into test and train data

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 1)


# GRID SEARCH

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Create logistic regression
classifier = LogisticRegression()

# Create regularization penalty space
penalty = ['l1', 'l2']

import numpy as np

# Create regularization hyperparameter space
C = np.logspace(0, 4, 10)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

# Create grid search using 5-fold cross validation
clf = GridSearchCV(classifier, hyperparameters, n_jobs = -1)

# Fit grid search
clf = clf.fit(X_train, y_train)


# View best hyperparameters
best_accuracy = clf.best_score_
best_parameters = clf._params_