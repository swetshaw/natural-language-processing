# -*- coding: utf-8 -*-
"""
Created on Sat May 15 22:05:45 2021

@author: shaws
"""

import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

paragraph = """Global warming is a hike in the average global temperature on earth. Burning of excess fossil fuel and the release of toxic fumes into the atmosphere is the major cause behind global warming. Global warming can have disastrous effects on living organisms. The impacts of global warming are widespread and unspecific. Some areas experience a sudden rise in the temperature while others witness a sudden fall in it. 
The major reason for global warming is the burning of fossil fuel for energy. It has been noted that the average temperature on the earth has increased by 1.5 degrees Celsius in the past decade. This is a reason for worry as it can damage ecosystems and might result in the disruption of the environment. Global warming can be checked if we take concrete steps towards restoring the lost vegetation in our forests. We can also use clean energy sources like wind energy, solar energy, and tidal energy to check the rise in global warming.
"""

lemmatizer = WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)

doc = []

for i in range(len(sentences)):
    clean_text = re.sub('[^a-zA-Z]', ' ', sentences[i])
    clean_text = clean_text.lower()
    clean_text = clean_text.split()
    clean_text = [lemmatizer.lemmatize(word) for word in clean_text if word not in set(stopwords.words('english'))]
    clean_text =  ' '.join(clean_text)
    doc.append(clean_text)
    
from sklearn.feature_extraction.text import TfidfVectorizer
tfid = TfidfVectorizer()
X = tfid.fit_transform(doc).toarray()
  
unique_words = tfid.get_feature_names()
