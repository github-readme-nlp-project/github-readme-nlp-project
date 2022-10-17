import pandas as pd
import numpy as np
import unicodedata
import re
import nltk
from nltk.corpus import stopwords

# potentially for project use
def clean_df(df, cols_to_clean, method='lemmatize', extra_words=[], exclude_words=[]):
    d = df.copy()
    for col in cols_to_clean:
        d[col+'_clean'] = clean_data(d[col], method, extra_words, exclude_words)
    return d

def clean_data(col, method='lemmatize', extra_words=[], exclude_words=[]):
    bc = [basic_clean(entry) for entry in col]   
    t = [tokenize(entry) for entry in bc]
    if method == 'lemmatize':
        l = [lemmatize(entry) for entry in t]
        return [remove_stopwords(entry, extra_words, exclude_words) for entry in l]
    if method == 'stemming':
        s = [stem(entry) for entry in t]
        return [remove_stopwords(entry, extra_words, exclude_words) for entry in s]


# backgroud functions used inside of above functions
def basic_clean(text):
    temp = text.lower()
    temp = unicodedata.normalize('NFKD', temp).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    temp = re.sub(r'[^a-zA-Z0-9\'\s]', '', temp)
    return temp

def tokenize(text):
    token = nltk.tokenize.ToktokTokenizer()
    temp = token.tokenize(text, return_str=True)
    return temp

def stem(text):
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in text.split(' ')]
    stemmed = ' '.join(stems)
    return stemmed

def lemmatize(text):
    wnl = nltk.stem.WordNetLemmatizer()
    lemons = [wnl.lemmatize(word) for word in text.split(' ')]
    lemmatized = ' '.join(lemons)
    return lemmatized

def remove_stopwords(text, extra_words=[], exclude_words=[]):
    words = text.split(' ')
    stopwords_list = stopwords.words('english')
    if len(extra_words) > 0:
        stopwords_list.extend(extra_words)
    if len(exclude_words) > 0:
        [stopwords_list.remove(w) for w in exclude_words]
    filtered_words = [word for word in words if word not in stopwords_list]
    return ' '.join(filtered_words)