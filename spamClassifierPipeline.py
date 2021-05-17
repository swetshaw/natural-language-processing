# -*- coding: utf-8 -*-
"""
Created on Mon May 17 16:36:51 2021

@author: shaws
"""

import pandas as pd

sms = pd.read_csv(r'SMSSpamCollection', sep='\t', names=['label', 'message'])

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

lemmatizer = WordNetLemmatizer()

doc = []

for i in range(len(sms)):
    clean_text = re.sub('[^a-zA-Z]', ' ', sms['message'][i])
    clean_text = clean_text.lower()
    clean_text = clean_text.split()
    clean_text = [lemmatizer.lemmatize(word) for word in clean_text if word not in set(stopwords.words('english'))]
    clean_text = ' '.join(clean_text)
    doc.append(clean_text)

label_map = {
    'spam': 1,
    'ham': 0
    }

y = sms['label'].map(label_map)    

X_train, X_test, y_train, y_test = train_test_split(doc, y, test_size = 0.2, random_state=0)
clf_pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])
clf_pipeline.fit(X_train, y_train)
predictions = clf_pipeline.predict(X_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions).round(2))

print(clf_pipeline.predict(["Congratulations, you have won a cash prize"]))
print(clf_pipeline.predict(["Come let's go for party"]))

