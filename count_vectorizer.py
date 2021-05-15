# -*- coding: utf-8 -*-
"""
Created on Sat May 15 19:13:54 2021

@author: shaws
"""
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

paragraph = """Global warming is a hike in the average global temperature on earth. Burning of excess fossil fuel and the release of toxic fumes into the atmosphere is the major cause behind global warming. Global warming can have disastrous effects on living organisms. The impacts of global warming are widespread and unspecific. Some areas experience a sudden rise in the temperature while others witness a sudden fall in it. 
The major reason for global warming is the burning of fossil fuel for energy. It has been noted that the average temperature on the earth has increased by 1.5 degrees Celsius in the past decade. This is a reason for worry as it can damage ecosystems and might result in the disruption of the environment. Global warming can be checked if we take concrete steps towards restoring the lost vegetation in our forests. We can also use clean energy sources like wind energy, solar energy, and tidal energy to check the rise in global warming.
"""

sentences = nltk.sent_tokenize(paragraph)
lemmatizer = WordNetLemmatizer()

doc = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    doc.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(doc).toarray()

words_list = cv.get_feature_names()
