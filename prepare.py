import pandas as pd
import numpy as np
import unicodedata
import re
import nltk
from nltk.corpus import stopwords
import nltk.sentiment
from sklearn.model_selection import train_test_split

def split_data(df):
    '''
    This function takes in the data as a dataframe and splits it into 
    train, validate, and test datasets with a 60/20/20 split.
    
    It returns the train, validate, and test datasets as dataframes.
    '''
    # create train and test (80/20 split) from the orginal dataframe
    train, test = train_test_split(df, test_size=0.2, random_state=123)
    # create train and validate (75/25 split) from the train dataframe
    train, val = train_test_split(train, test_size=.25, random_state=123)
    # return the datasets
    return train, val, test


def clean_df(df, cols_to_clean, method='lemmatize', extra_words=[], exclude_words=[]):
    '''
    This function takes in:
        - data as a dataframe
        - list of columns to be cleaned,
        - selection of method to clean the data (lemmatize or stemming)
        - list of extra words to be added to the stopwords to be removed
        - list of words to remove from the stopwords list
    
    It drops the rows with nulls, puts the language feature into bins to consolidate the
    lower count languages into 'Other', cleans the data via the method input while removing 
    or adding stopwords as directed, and finally feature engineers rm_length as a word count
    and sentiment as the compound value of each readme.
    '''
    # create a copy to preserve the original
    d = df.copy()
    # drop all rows with nulls
    d = d.dropna()
    # call create_other to bin the languages
    d.language = create_other(d)
    # for each column in cols_to_clean
    for col in cols_to_clean:
        # create a new column in the dataframe with the clean version of the column
        d[col+'_clean'] = clean_data(d[col], method, extra_words, exclude_words)
    # feature engineer rm_length to hold the word count    
    d['rm_length'] = [len(text) for text in d.readme_contents_clean]
    # create the sentiment analyzer
    sia = nltk.sentiment.SentimentIntensityAnalyzer()
    # feature engineer sentiment to hold the compound sentiment score
    d['sentiment'] = d.readme_contents_clean.apply(lambda doc: sia.polarity_scores(doc)['compound'])
    # return the dataframe
    return d

def create_other(df):
    '''
    This function takes in the data as a dataframe and bins languages down to:
    - JavaScript
    - Python
    - Java
    - Ruby
    - Other (holds all other languages)
    '''
    # create a copy of the language series to preserve the original dataframe
    copy = df.language.copy()
    # create an empty list to eventually hold what will be returned
    output = []
    # for every language in the copy
    for lang in copy:
        # if the language is not JavaScript, Python, Java or Ruby
        if (lang != 'JavaScript') & (lang != 'Python') & (lang != 'Java') & (lang != 'Ruby'):
            # save it as Other
            output.append('Other')
        else:
            # otherwise save it as itself
            output.append(lang)
    # return the final list with the adjusted language titles
    return output

def clean_data(col, method='lemmatize', extra_words=[], exclude_words=[]):
    '''
    This function takes: 
        - the column to be cleaned
        - the method (lemmatize or stemming)
        - any extra words to be added to stopwords
        - any words to be taken out of stopwords
    '''
    # call basic_clean on every entry in the series
    bc = [basic_clean(entry) for entry in col]   
    # call tokenize on every entry in the basic clean returned series
    t = [tokenize(entry) for entry in bc]
    # if the method selected is lemmatize...
    if method == 'lemmatize':
        # call lemmatize on every entry in the tokenized returned series
        l = [lemmatize(entry) for entry in t]
        # return after removing stopwords from every entry in the lemmatized series
        return [remove_stopwords(entry, extra_words, exclude_words) for entry in l]
    # if the method selected is stemming
    if method == 'stemming':
        # call stem on every entry in the tokenized series
        s = [stem(entry) for entry in t]
        # return after removing stopwords from every entry in the stemmed series
        return [remove_stopwords(entry, extra_words, exclude_words) for entry in s]


# backgroud functions used inside of above functions
def basic_clean(text):
    '''
    This function takes in text as a string and cleans it via:
        - making it all lowercase
        - converting all characters to be of the same keyboard (no ~ or ` for example)
        - removing anything that is not a letter, number or apostophe
    '''
    # make it lowercase
    temp = text.lower()
    # convert characters to be of the same keyboard
    temp = unicodedata.normalize('NFKD', temp).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # remove all characters that are not a letter, number, or apostrophe
    temp = re.sub(r'[^a-zA-Z0-9\'\s]', '', temp)
    # return the clean text
    return temp

def tokenize(text):
    '''
    This function takes in text as a string and returns it tokenized.
    '''
    # make the tokenizer
    token = nltk.tokenize.ToktokTokenizer()
    # use the tokenizer on the text
    temp = token.tokenize(text, return_str=True)
    # return the tokenized text
    return temp

def stem(text):
    '''
    This function takes in text as a string and uses the PorterStemmer() from
    nltk on it.
    '''
    # make the stemmer
    ps = nltk.porter.PorterStemmer()
    # stem each word in the text
    stems = [ps.stem(word) for word in text.split(' ')]
    # rejoin the stemmed words together
    stemmed = ' '.join(stems)
    # return the stemmed text
    return stemmed

def lemmatize(text):
    '''
    This function takes in text as a string and uses the WordNetLemmatizer()
    from nltk on it.
    '''
    # make the lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    # lemmatize each word in the text
    lemons = [wnl.lemmatize(word) for word in text.split(' ')]
    # rejoin the lemmatized words
    lemmatized = ' '.join(lemons)
    # return the lemmatized text
    return lemmatized

def remove_stopwords(text, extra_words=[], exclude_words=[]):
    '''
    This function takes in text as a string as well as lists for words to 
    add or remove from the stopwords list. It then returns the text with
    all the stopwords removed.
    '''
    # split the text into a list of words
    words = text.split(' ')
    # set up the stopwords list
    stopwords_list = stopwords.words('english')
    # if any extra_words were passed..
    if len(extra_words) > 0:
        # add them to the stopwords list
        stopwords_list.extend(extra_words)
    # if any exclude_words were passed...
    if len(exclude_words) > 0:
        # removed them from the stopwords list
        [stopwords_list.remove(w) for w in exclude_words]
    # remove the words in the text if they are in the stopwords list
    filtered_words = [word for word in words if word not in stopwords_list]
    # return the text without the stopwords
    return ' '.join(filtered_words)