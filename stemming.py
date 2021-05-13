# -*- coding: utf-8 -*-
"""
Created on Fri May 14 02:15:44 2021

@author: shaws

Here we will be seeing how to apply the process of stemming to get the stem word. 
We have applied stemming using two ways of tokenizing the paragraph:
    - tokenizing sentences
    - tokenizing words
"""

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

paragraph = """Global warming is a hike in the average global temperature on earth. Burning of excess fossil fuel and the release of toxic fumes into the atmosphere is the major cause behind global warming. Global warming can have disastrous effects on living organisms. The impacts of global warming are widespread and unspecific. Some areas experience a sudden rise in the temperature while others witness a sudden fall in it. 
The major reason for global warming is the burning of fossil fuel for energy. It has been noted that the average temperature on the earth has increased by 1.5 degrees Celsius in the past decade. This is a reason for worry as it can damage ecosystems and might result in the disruption of the environment. Global warming can be checked if we take concrete steps towards restoring the lost vegetation in our forests. We can also use clean energy sources like wind energy, solar energy, and tidal energy to check the rise in global warming.
"""

# Tokenize sentences
sentences = nltk.sent_tokenize(paragraph)
stemmer = PorterStemmer()

for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)
    
# Tokenize words    
words = nltk.word_tokenize(paragraph)

cleaned_text = []
for word in words:
    if word not in set(stopwords.words('english')):
        cleaned_text.append(stemmer.stem(word))
print(' '.join(cleaned_text))