import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import sklearn.metrics as met

def run_models(train, val):
    '''
    This function takes in the train and validate datasets, creates a
    TfidVectorizer and runs the following models:
        - Baseline
        - Decision Tree
        - Random Forest
        - Multinomial Naive Bayes
    
    and returns the train and validate accuracy results in a dataframe.
    '''
    # create X and y for train and validate
    X_train, y_train = train.readme_contents_clean, train.language
    X_val, y_val = val.readme_contents_clean, val.language
    
    # set baseline as the mode language
    baseline = train.language.value_counts().idxmax()
    # create a datafrome with the actual languages
    hold = pd.DataFrame(train.language)
    # rename it as actual for easy reading
    hold.columns = ['actual']
    # create a column for the baseline value
    hold['baseline_pred'] = baseline
    
    # create an empty dictionary to hold model results
    results = {}
    
    # add the baseline accuracy to the results dictionary 
    results['baseline'] = {
        'train_acc':met.accuracy_score(hold.actual, hold.baseline_pred)
    }

    # create the tfidvectorizer
    tv = TfidfVectorizer()
    # fit and transform X_train into a bag of words
    train_bow = tv.fit_transform(X_train)
    # transform X_val into a bag of words
    val_bow = tv.transform(X_val)

    # create a decision tree
    tree = DecisionTreeClassifier(max_depth=5, random_state=123)
    # fit the tree on the train bag of words and y_train
    tree.fit(train_bow, y_train)
    # add the decision tree accuracy to the results dictionary 
    results['Decision Tree'] = {
        'train_acc':tree.score(train_bow, y_train),
        'val_acc':tree.score(val_bow, y_val)
    }

    # create a random forest
    forest = RandomForestClassifier(max_depth=5, random_state=123)
    # fit the forest on the train bag of words and y_train
    forest.fit(train_bow, y_train)
    # add the random forest accuracy to the results dictionary
    results['Random Forest'] = {
        'train_acc':forest.score(train_bow, y_train),
        'val_acc':forest.score(val_bow, y_val)
    }

    # create a naive bayes model
    nb = MultinomialNB(alpha=1)
    # fit the model on the train bag of words and y_train
    nb.fit(train_bow, y_train)
    # add the model's accuracy to the results dictionary
    results['Naive Bayes'] = {
        'train_acc':nb.score(train_bow, y_train),
        'val_acc':nb.score(val_bow, y_val)
    }
    # return the results dictionary as a dataframe
    return pd.DataFrame(results).T

def run_best(train, val, test):
    '''
    This function takes in the train, validate, and test datasets and 
    runs the decision tree classifier on it and returns the results in
    a dataframe.
    '''
    # set up X and y for train, validate, and test
    X_train, y_train = train.readme_contents_clean, train.language
    X_val, y_val = val.readme_contents_clean, val.language
    X_test, y_test = test.readme_contents_clean, test.language
    
    # create the tfidvectorizer
    tv = TfidfVectorizer()
    # use the tfidvecotrizer to create a bag of words for train, validate, and test
    train_bow = tv.fit_transform(X_train)
    val_bow = tv.transform(X_val)
    test_bow = tv.transform(X_test)

    # create an empty dictionary to hold the results data
    final = {}

    # create a decision tree
    tree = DecisionTreeClassifier(max_depth=5, random_state=123)
    # fit the tree to the train bag of words and y_train
    tree.fit(train_bow, y_train)
    # add the tree's accuracy results to the dictionary
    final['Decision Tree'] = {
        'train_acc':tree.score(train_bow, y_train),
        'val_acc':tree.score(val_bow, y_val),
        'test_acc':tree.score(test_bow, y_test)
    }
    # return the dictionary holding the results as a dataframe
    return pd.DataFrame(final)