import pandas as pd
import numpy as np
import nltk
import re
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from nltk.stem.snowball import FrenchStemmer


# using stemming of words
stemmer = FrenchStemmer()

# spliting a string using a regular expression 
tokenizer = nltk.RegexpTokenizer(r'\w+')

# lists of stop words to ignore
stop_fr  = nltk.corpus.stopwords.words('french')
stop_spacy_fr = list(fr_stop)

def parse_text(text):
    """
    Parse text to make it ready for further NLP analysis:
    lower case, remove digits, some characters, stop words...
    """ 
    text = [re.sub("\d+", "", word) for word in text]
    text = [re.sub(r'[\(\[\)\]\{\}\.\/]+', '', word) for word in text]
    text = [word.lower() for word in text]
    text = tokenizer.tokenize(str(text))
    text = [word for word in text if not word in stop_fr] 
    text = [word for word in text if not word in stop_spacy_fr]
    
    return text
