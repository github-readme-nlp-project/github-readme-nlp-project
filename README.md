
## Predicting Repository Languages
                      
### See our presentation [here](https://www.canva.com/design/DAFPWnqLY2U/RioilctzSLT2MfJknT7I4g/view?utm_content=DAFPWnqLY2U&utm_campaign=designshare&utm_medium=link&utm_source=recording_view) or just the slides [here](https://www.canva.com/design/DAFPWnqLY2U/ig8CXwBMyThosppiCDUu1g/edit?utm_content=DAFPWnqLY2U&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

### Project Goals:
The goal of this project are analyse the contents of README and to build a model that can predict the main programming language of a repository.


### Project Description
The goal of this project is to build a model that can predict the main programming language of from README file of Github repository. This project started by scraping README files from various GitHub repositories using web scraping techniques. Following the acquisition and preparation of our data, our team used natural language processing exploration methods such as word clouds, bigrams, and trigrams. We used multiclass classification methods to create multiple machine learning models.
The end goal was to create an NLP model that accurately predicted the programming language used in a github repository based on the words and word combinations found in the readme files.

### Initial Questions
- What are the most common words between languages?
- Which combination of words appear the most frequent?
- Does the length of the README vary by programming language?
- How do languages differ in sentiment within their README?

### Data Dictionary

Target
|Name|Type|Description|
|-----|------------|---|
|language|str|The primary coding language of the repository|


Features
|Name|Type|Description|
|-----|------------|---|
|repo	|str|	the name of repository
|readme_contents|str	|The raw README text|
|readme_contents_clean|str|The cleaned and lemmatized README text|
|rm_length	|int	|the word count of the clean README text|
|sentiment|	float	|the compound sentiment value of the clean README text|


### Steps to Replicate this project
You will need the following tools/libraries
- python
- pandas
- numpy
- sci-kit learn
- matplotlib
- seaborn
- NLTK
- BeautifulSoup

Steps to recreate
- to clone this repo, on your terminal, type git clone https://github.com/github-readme-nlp-project/github-readme-nlp-project
- download prepare.py, explore.py and model.py. into your directory if you are not cloning
- download the data.json file to use the same dataset from the analysis or use acquire.py file
- run the final notebook





### Project Outline:
### Acquisiton of data:
- Conducted web scraping of 1000 repositories' readme contents as of 17th October, 2022.
- Filters used were English language, and programming language (JavaScript,Java,Python,Ruby, and C++)
- Saved the data locally in JSON format.

### Prepare and clean data
- Dropped nulls
- Lowercased all text
- Normalized, encoded, and decoded the text
- Tokenized the data using NTLK Toktoktokenizer
- Lemmatized the data using ntlk Wordnetlemmatizer
- Removed stopwords using NTLK standard english list and extra words
- Splitted data in train (60%) , validate(20%), test(20%)


### Exploration
- Target variable was Language
- Used train data only for exploration and use basic exploration data analysis techniques.
- Created a language bin that empasses languages which are not JavaScript,Java,Python or Ruby
- Using various visualizations such as boxplot, wordcloud, bigrams, sentiment analysis plot, explored answers to the initial questions

### Modeling
- Created and evaluated various classification models, varying the data preprocessing methods (Count vectorizier, TF- IDF) and associated hyperparameters.
- Baseline prediction was Java, with baseline accuracy of 23%.
- Models used were DecisionTree, RandomForest and Naive Bayes.
- Decision Tree model performed the best at 69% accuracy

### Conclusion and Recommendation
- We succedded on analysing the contents of README and building a model that can predict the main programming language.
- Ruby projects were overwhelmingly positive in compound sentiment while also being the leader in README lengths; Java was the opposite for both.
- Overall common words were used equally across all languages.
- We recommend exploring within a single coding language and look at what is in READMEs across various human language which may provide insight into how different regional coders use that coding language.
- We recommend researching more on shorter README's as they tend towards negative sentiment.

### Next Steps
- Given more time, further cleaning of the README content could hone in the model results. (Ex: removing urls and other markdown syntax)

