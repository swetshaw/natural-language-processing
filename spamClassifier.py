# -*- coding: utf-8 -*-
"""
Created on Mon May 17 01:06:55 2021

@author: shaws
"""

import pandas as pd

sms = pd.read_csv(r'SMSSpamCollection', sep='\t', names=['label', 'message'])

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

doc = []

for i in range(len(sms)):
    clean_text = re.sub('[^a-zA-Z]', ' ', sms['message'][i])
    clean_text = clean_text.lower()
    clean_text = clean_text.split()
    clean_text = [lemmatizer.lemmatize(word) for word in clean_text if word not in set(stopwords.words('english'))]
    clean_text = ' '.join(clean_text)
    doc.append(clean_text)
    
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(doc).toarray()

label_map = {
    'spam': 1,
    'ham': 0
    }

y = sms['label'].map(label_map)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

model = nb.fit(X_train, y_train)

y_pred = model.predict(X_test)


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred).round(2))



