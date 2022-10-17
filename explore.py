import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from wordcloud import WordCloud

def q1_vis(all_words, word_counts):
    plt.figure(figsize=(16, 9))

    (word_counts.sort_values('All', ascending=False)
     .head(20)
     .apply(lambda row: row/row['All'], axis = 1)
     .drop(columns = 'All')
     .sort_values(by = 'JavaScript')
     .plot.barh(stacked = True, width = 1, ec = 'k')
    )
    plt.title('% of lang for the most common 20 words')
    plt.show()
    print()
    plt.figure(figsize=(12,8))
    img = WordCloud().generate(all_words)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Most Common README Words')
    plt.show()
    
    
def q1b_vis(all_words):
    plt.figure(figsize=(12,8))
    pd.Series(nltk.ngrams(all_words.split(),2)).value_counts().sort_values().tail(20).plot.barh()
    plt.title('20 Most Common Bigrams in READMEs')
    plt.xlabel('Occurances')
    plt.show()
    
def q2_vis(train):
    plt.figure(figsize=(12,8))
    sns.boxplot(data=train, x='language', y='rm_length')
    plt.title('Python Lags in Word Count where JavaScript Leads')
    plt.ylabel('README Word Count')
    plt.ylim(0,50000)
    plt.show()
    
def q3_vis(train):
    plt.figure(figsize=(12,8))
    sns.kdeplot(train[train.language == 'JavaScript'].sentiment, label='JavaScript')
    sns.kdeplot(train[train.language == 'Python'].sentiment, label='Python')
    sns.kdeplot(train[train.language == 'Go'].sentiment, label='Go')
    sns.kdeplot(train[train.language == 'Other'].sentiment, label='Other')
    plt.legend(['JavaScript', 'Python', 'Go', 'Other'])
    plt.xlim(-1,1)
    plt.title('JavaScript uses Highly Positive Language')
    plt.show()
    
    print()
    
    plt.figure(figsize=(12,8))
    sns.kdeplot(train[train.language == 'JavaScript'].rm_length,train[train.language == 'JavaScript'].sentiment, levels = 30, shade = True )
    sns.kdeplot(train[train.language == 'Python'].rm_length,train[train.language == 'Python'].sentiment, levels = 30, shade = True, alpha = 0.5 )
    plt.xlabel('README Word Count')
    plt.title('Python READMEs vary more than JavaScript in Word Count and Sentiment')
    plt.show()
    