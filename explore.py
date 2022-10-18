import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from wordcloud import WordCloud
import scipy.stats as stats


def q2_vis(all_words, word_counts):
    plt.figure(figsize=(12,8))
    img = WordCloud().generate(all_words)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Most Common README Words')
    plt.show()
    
    print()
    
    (word_counts.sort_values('All', ascending=False)
     .head(20)
     .apply(lambda row: row/row['All'], axis = 1)
     .drop(columns = 'All')
     .sort_values(by = 'JavaScript')
     .plot.barh(stacked = True, width = 1, ec = 'k', figsize=(12, 8))
    )
    plt.title('% of lang for the most common 20 words')
    plt.show()
    
def q2b_vis(all_words):
    plt.figure(figsize=(12,8))
    pd.Series(nltk.ngrams(all_words.split(),2)).value_counts().sort_values().tail(20).plot.barh()
    plt.title('20 Most Common Bigrams in READMEs')
    plt.xlabel('Occurances')
    plt.show()
    
def q1_vis(train):
    plt.figure(figsize=(12,8))
    sns.boxplot(data=train, x='language', y='rm_length', order=['JavaScript', 'Java', 'Python', 'Ruby', 'Other'])
    plt.title('Java Lags in Word Count where Ruby Leads')
    plt.ylabel('README Word Count')
    plt.ylim(0,35000)
    plt.show()
    
def q1_stats(train):
    μ = train.rm_length.mean()
    java_words = train[train.language == 'Java'].rm_length
    α = 0.05
    
    t, p = stats.ttest_1samp(java_words, μ)
    if (t < 0) & (p/2 < α):
        print(f'''Reject the Null Hypothesis. 
    
Findings suggest Java projects use less words on average in their READMEs
than projects in other languages.''')
    else:
         print(f'''Fail to reject the Null Hypothesis. 
     
Findings suggest Java projects use equal or more words on average in 
their READMEs than projects in other languages.''')
    
def q3_vis(train):
    plt.figure(figsize=(12,8))
    sns.kdeplot(train[train.language == 'JavaScript'].sentiment, label='JavaScript', color='blue')
    plt.axvline(train[train.language == 'JavaScript'].sentiment.mean(),.7, color='blue')

    sns.kdeplot(train[train.language == 'Java'].sentiment, label='Java', color= 'darkorange')
    plt.axvline(train[train.language == 'Java'].sentiment.mean(),.7, color='darkorange')

    sns.kdeplot(train[train.language == 'Python'].sentiment, label='Python', color='green')
    plt.axvline(train[train.language == 'Python'].sentiment.mean(),.7, color='green')

    sns.kdeplot(train[train.language == 'Ruby'].sentiment, label='Ruby', color='red')
    plt.axvline(train[train.language == 'Ruby'].sentiment.mean(),.7, color='red')

    sns.kdeplot(train[train.language == 'Other'].sentiment, label='Other', color='black')
    plt.axvline(train[train.language == 'Other'].sentiment.mean(),.7, color='black')

    plt.text(.22, 4, '''Average Compound Sentiment 
                            by Language''')
    
    plt.legend()
    plt.xlim(-1,1)
    plt.title('Ruby uses Highly Positive Language')
    plt.show()
    
    print()
    
    plt.figure(figsize=(12,8))
    sns.kdeplot(train[train.language == 'Java'].rm_length,train[train.language == 'Java'].sentiment,
                levels = 30, shade = True, label='Java', color='blue')
    sns.kdeplot(train[train.language == 'Ruby'].rm_length,train[train.language == 'Ruby'].sentiment,
                levels = 30, shade = True, alpha = 0.5, label='Ruby', color='orange')
    plt.xlabel('Word Count')
    plt.title('Ruby READMEs appear to be in Sentiment bins compared to Java READMEs')
    plt.legend()
    plt.show()
    
def q3_stats(train):
    α = 0.05
    ruby = train[train.language == 'Ruby'].sentiment
    java = train[train.language == 'Java'].sentiment

    s, pval = stats.levene(ruby, java)

    t, p = stats.ttest_ind(ruby, java, equal_var=(pval>α)) 

    if (t>0) & (p/2 < α):
        print(f'''Reject the Null Hypothesis.

Findings suggest the average sentiment for Ruby READMEs is
higher than the average sentiment for Java READMEs''')
    else:
        print(f'''Fail to reject the Null Hypothesis.

Findings suggest the average sentiment for Ruby READMEs is
equal or lower than the average sentiment for Java READMEs''')