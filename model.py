import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import sklearn.metrics as met

def run_models(train, val):
    X_train, y_train = train.readme_contents_clean, train.language
    X_val, y_val = val.readme_contents_clean, val.language
    
    baseline = train.language.value_counts().idxmax()
    hold = pd.DataFrame(train.language)
    hold.columns = ['actual']
    hold['baseline_pred'] = baseline
    results = {}

    results['baseline'] = {
        'train_acc':met.accuracy_score(hold.actual, hold.baseline_pred)
    }

    tv = TfidfVectorizer()
    train_bow = tv.fit_transform(X_train)
    val_bow = tv.transform(X_val)

    tree = DecisionTreeClassifier(max_depth=5, random_state=123)
    tree.fit(train_bow, y_train)
    results['Decision Tree'] = {
        'train_acc':tree.score(train_bow, y_train),
        'val_acc':tree.score(val_bow, y_val)
    }

    forest = RandomForestClassifier(max_depth=5, random_state=123)
    forest.fit(train_bow, y_train)
    results['Random Forest'] = {
        'train_acc':forest.score(train_bow, y_train),
        'val_acc':forest.score(val_bow, y_val)
    }

    nb = MultinomialNB(alpha=1)
    nb.fit(train_bow, y_train)
    results['Naive Bayes'] = {
        'train_acc':nb.score(train_bow, y_train),
        'val_acc':nb.score(val_bow, y_val)
    }

    return pd.DataFrame(results).T

def run_best(train, val, test):
    X_train, y_train = train.readme_contents_clean, train.language
    X_val, y_val = val.readme_contents_clean, val.language
    X_test, y_test = test.readme_contents_clean, test.language
    
    tv = TfidfVectorizer()
    train_bow = tv.fit_transform(X_train)
    val_bow = tv.transform(X_val)
    test_bow = tv.transform(X_test)

    final = {}

    tree = DecisionTreeClassifier(max_depth=5, random_state=123)
    tree.fit(train_bow, y_train)
    final['Decision Tree'] = {
        'train_acc':tree.score(train_bow, y_train),
        'val_acc':tree.score(val_bow, y_val),
        'test_acc':tree.score(test_bow, y_test)
    }

    return pd.DataFrame(final)