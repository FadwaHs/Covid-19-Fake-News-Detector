# -*- coding: utf-8 -*-
"""
Created on Mon May 17 13:46:40 2021

@author: fadwa
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn import tree
import pickle



covid= pd.read_csv("data.csv")

del covid['label']
del covid['text']
covid.drop(covid.columns[0], axis=1, inplace=True)


covid.isnull().values.any()
covid_sans_NAN = covid.dropna()
covid_sans_NAN.isnull().values.any()


features = covid_sans_NAN['title']
labels = covid_sans_NAN['subcategory']

X=features
y=labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

tfvect = TfidfVectorizer(stop_words='english',max_df=0.7)

tfid_x_train = tfvect.fit_transform(X_train)
tfid_x_test = tfvect.transform(X_test)


modeltree = tree.DecisionTreeClassifier(criterion='entropy')
modeltree.fit(tfid_x_train, y_train)

y_predict = modeltree.predict(tfid_x_test)
#score
from sklearn.metrics import accuracy_score
print ("score de classification=", accuracy_score(y_test, y_predict))#forte



pickle.dump(modeltree,open('modelcovid.pkl', 'wb'))

# load the model from disk
loaded_model = pickle.load(open('modelcovid.pkl', 'rb'))

    
    
def fake_news_det1(news):
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    print(prediction)


fake_news_det1("Corona virus was first discovered in wihan")













