# -*- coding: utf-8 -*-
"""
Created on Tue May 18 10:50:20 2021

@author: shaws
"""
# import pandas to read the data
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r'Daily news stock prediction/Combined_News_DJIA.csv')

# check for null values
df.isnull().sum()

# remove rows with null values
df.dropna(inplace= True)

import seaborn as sns
sns.countplot(df['Label'])

# split into train and test dataset
df_train = df[df['Date'] < '20150101']
df_test = df[df['Date'] > '20141231']



def clean_data(dataset):
    data = dataset.iloc[:,2:27]
    data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)
    return data


def combine_data(data):
    headlines = []
    for i in range(len(data.index)):
        headlines.append(' '.join(str(x) for x in data.iloc[i, :]))
    return headlines

def lemmatize_data(data, lemmatizer):
    cleaned_dataset = []
    for i in range(len(data)):
        clean_text = data[i].lower()
        clean_text = clean_text.split()
        clean_text = [lemmatizer.lemmatize(word) for word in clean_text if word not in stopwords.words('english')]
        cleaned_dataset.append(' '.join(clean_text))
    return cleaned_dataset

def vectorize_data(data, cv):
    vectorized_dataset = cv.fit_transform(data)
    return vectorized_dataset


# clean train and test data
clean_train_data = clean_data(df_train)
clean_test_data = clean_data(df_test)

# combine the headllines in single column
comb_train_data = combine_data(clean_train_data)
comb_test_data = combine_data(clean_test_data)

lemmatizer = WordNetLemmatizer()

# lemmatize data
train_data = lemmatize_data(comb_train_data, lemmatizer)
test_data = lemmatize_data(comb_test_data, lemmatizer)

cv = CountVectorizer(ngram_range=(2,2))

# vectorize data
vec_train_data = vectorize_data(train_data, cv)
vec_test_data = cv.transform(test_data)

# create classifier
rf_clf = RandomForestClassifier(n_estimators=200, criterion='entropy')
rf_clf.fit(vec_train_data, df_train['Label'])

# run precictions on test data
y_pred = rf_clf.predict(vec_test_data)

# check accuracy and classification report
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
print(classification_report(df_test['Label'], y_pred))
accuracy_score(df_test['Label'], y_pred)
confusion_matrix(df_test['Label'], y_pred)

pd.crosstab(df_test["Label"], y_pred, rownames=["Actual"], colnames=["Predicted"])




