import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from wordcloud import WordCloud
import scipy.stats as stats


def q2_vis(all_words, word_counts):
    '''
    This function displays the visualizations for question 2 of the
    the GitHub README NLP project.
    '''
    # make it big
    plt.figure(figsize=(12,8))
    # create the word cloud
    img = WordCloud().generate(all_words)
    # show the wordcloud
    plt.imshow(img)
    # remove the axis
    plt.axis('off')
    # give it a title
    plt.title('Most Common README Words')
    # show the final product
    plt.show()
    
    # print a line for space between visuals
    print()
    
    # sort the word count values by the all column
    (word_counts.sort_values('All', ascending=False)
     # only show the top 20
     .head(20)
     # get the fraction it shows up in that language vs all languages
     .apply(lambda row: row/row['All'], axis = 1)
     # drop the all column
     .drop(columns = 'All')
     # sort by the javascript column
     .sort_values(by = 'JavaScript')
     # make it a horizontal bar graph
     .plot.barh(stacked = True, width = 1, ec = 'k', figsize=(12, 8))
    )
    # give it a title
    plt.title('% of lang for the most common 20 words')
    # show the final product
    plt.show()
    
    
    
def q2b_vis(all_words):
    '''
    This function displays the visualizations for question 2b of the
    the GitHub README NLP project.
    '''
    # make it big
    plt.figure(figsize=(12,8))
    # plot a horizontal bar graph of the top 20 bigrams in all_words
    pd.Series(nltk.ngrams(all_words.split(),2)).value_counts().sort_values().tail(20).plot.barh()
    # give it a title
    plt.title('20 Most Common Bigrams in READMEs')
    # fix the x-axis label
    plt.xlabel('Occurances')
    # show the final product
    plt.show()
    
    
    
def q1_vis(train):
    '''
    This function displays the visualizations for question 1 of the
    the GitHub README NLP project.
    '''
    # make it big
    plt.figure(figsize=(12,8))
    # create a boxplot showing word count vs language, 
    # fix the order so the colors line up with other plots
    sns.boxplot(data=train, x='language', y='rm_length', order=['JavaScript', 'Java', 'Python', 'Ruby', 'Other'])
    # give it a title
    plt.title('Java Lags in Word Count where Ruby Leads')
    # fix the y-axis label
    plt.ylabel('README Word Count')
    # zoom in a little to display the differences better
    plt.ylim(0,35000)
    # show the final product
    plt.show()
    
    
    
def q1_stats(train):
    '''
    This function displays the statistical results for question 1 of the
    the GitHub README NLP project.
    '''
    # calculate the population mean of word count
    μ = train.rm_length.mean()
    # create the sample of Java word counts
    java_words = train[train.language == 'Java'].rm_length
    # set alpha
    α = 0.05
    # run the ttest for one sample, one tail test
    t, p = stats.ttest_1samp(java_words, μ)
    # if t is negative and half of p is less than alpha
    if (t < 0) & (p/2 < α):
        # print this result
        print(f'''Reject the Null Hypothesis. 
    
Findings suggest Java projects use less words on average in their READMEs
than projects in other languages.''')
    else:
        # otherwise print this result
         print(f'''Fail to reject the Null Hypothesis. 
     
Findings suggest Java projects use equal or more words on average in 
their READMEs than projects in other languages.''')
    
    
    
def q3_vis(train):
    '''
    This function displays the visualizations for question 3 of the
    the GitHub README NLP project.
    '''
    # make it big
    plt.figure(figsize=(12,8))
    # plot the sentiment densities for JavaScript and a line for its average
    sns.kdeplot(train[train.language == 'JavaScript'].sentiment, label='JavaScript', color='blue')
    plt.axvline(train[train.language == 'JavaScript'].sentiment.mean(),.7, color='blue')
    # plot the sentiment densities for Java and a line for its average
    sns.kdeplot(train[train.language == 'Java'].sentiment, label='Java', color= 'darkorange')
    plt.axvline(train[train.language == 'Java'].sentiment.mean(),.7, color='darkorange')
    # plot the sentiment densities for Python and a line for its average
    sns.kdeplot(train[train.language == 'Python'].sentiment, label='Python', color='green')
    plt.axvline(train[train.language == 'Python'].sentiment.mean(),.7, color='green')
    # plot the sentiment densities for Ruby and a line for its average
    sns.kdeplot(train[train.language == 'Ruby'].sentiment, label='Ruby', color='red')
    plt.axvline(train[train.language == 'Ruby'].sentiment.mean(),.7, color='red')
    # plot the sentiment densities for Other and a line for its average
    sns.kdeplot(train[train.language == 'Other'].sentiment, label='Other', color='black')
    plt.axvline(train[train.language == 'Other'].sentiment.mean(),.7, color='black')
    # add a block of text to explain the lines showing the averages for each language
    plt.text(.22, 4, '''Average Compound Sentiment 
                            by Language''')
    # add a legend
    plt.legend()
    # zoom in a bit
    plt.xlim(-1,1)
    # give it a title
    plt.title('Ruby uses Highly Positive Language')
    # show the final product
    plt.show()
    
    # print a line to put space between visuals
    print()
    
    # make it big
    plt.figure(figsize=(12,8))
    # plot the densities for Java sentiment and word count
    sns.kdeplot(train[train.language == 'Java'].rm_length,train[train.language == 'Java'].sentiment,
                levels = 30, shade = True, label='Java', color='blue')
    # plot the densities for Ruby sentiment and word count
    sns.kdeplot(train[train.language == 'Ruby'].rm_length,train[train.language == 'Ruby'].sentiment,
                levels = 30, shade = True, alpha = 0.5, label='Ruby', color='orange')
    # fix the x-axis label
    plt.xlabel('Word Count')
    # give it a title
    plt.title('Ruby READMEs appear to be in Sentiment bins compared to Java READMEs')
    # put a legend on it
    plt.legend()
    # show the final product
    plt.show()
    
    
    
def q3_stats(train):
    '''
    This function displays the statistical results for question 3 of the
    the GitHub README NLP project.
    '''
    # set alpha
    α = 0.05
    # create the ruby sentiment sample
    ruby = train[train.language == 'Ruby'].sentiment
    # create the java sentiment sample
    java = train[train.language == 'Java'].sentiment
    # test for equal variances
    s, pval = stats.levene(ruby, java)
    # do the two sample, one tail ttest
    t, p = stats.ttest_ind(ruby, java, equal_var=(pval>α)) 
    # if t is positive and half of p is less than alpha
    if (t>0) & (p/2 < α):
        # print this result
        print(f'''Reject the Null Hypothesis.

Findings suggest the average sentiment for Ruby READMEs is
higher than the average sentiment for Java READMEs''')
    else:
        # otherwise print this result
        print(f'''Fail to reject the Null Hypothesis.

Findings suggest the average sentiment for Ruby READMEs is
equal or lower than the average sentiment for Java READMEs''')