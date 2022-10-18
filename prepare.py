import pandas as pd
import numpy as np
import unicodedata
import re
import nltk
from nltk.corpus import stopwords
import nltk.sentiment
from sklearn.model_selection import train_test_split

def split_data(df):
    train, test = train_test_split(df, test_size=0.2, random_state=123)
    train, val = train_test_split(train, test_size=.25, random_state=123)
    
    return train, val, test


def clean_df(df, cols_to_clean, method='lemmatize', extra_words=[], exclude_words=[]):
    d = df.copy()
    d = d.dropna()
    d.language = create_other(d)
    for col in cols_to_clean:
        d[col+'_clean'] = clean_data(d[col], method, extra_words, exclude_words)
    d['rm_length'] = [len(text) for text in d.readme_contents_clean]
    sia = nltk.sentiment.SentimentIntensityAnalyzer()
    d['sentiment'] = d.readme_contents_clean.apply(lambda doc: sia.polarity_scores(doc)['compound'])
    return d

def create_other(df):
    copy = df.language.copy()
    output = []
    for lang in copy:
        if (lang != 'JavaScript') & (lang != 'Python') & (lang != 'Java') & (lang != 'Ruby'):
            output.append('Other')
        else:
            output.append(lang)
    return output

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