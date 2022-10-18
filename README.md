## Project Goals:
The goal of this project are analyse the contents of README and to build a model that can predict the main programming language of a repository.


## Project Description
The goal of this project is to build a model that can predict the main programming language of from README file of Github repository. This project started by scraping README files from various GitHub repositories using web scraping techniques. Following the acquisition and preparation of our data, our team used natural language processing exploration methods such as word clouds, bigrams, and trigrams. We used multiclass classification methods to create multiple machine learning models.
The end goal was to create an NLP model that accurately predicted the programming language used in a github repository based on the words and word combinations found in the readme files.

## Initial Questions
- What are the most common words between languages?
- Which combination of words appear the most frequent?
- Does the length of the README vary by programming language?
- How do languages differ in sentiment within their README?

## Data Dictionary

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


## Steps to Replicate this project
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





## Project Outline:
### Acquisiton of data:
- Conduct web scraping of repositories' readme contents.
- 
### Prepare and clean data
- Drop nulls
- Lowercase all text
- Normalize, encode, and decode the text
- Remove stopwords using NTLK standard english list
- Tokenize the data
- Stem and Lemmatize
- Split data in train (60%) , validate(20%), test(20%)


### Exploration

- bigrams and trigrams for top language
- Analyze bigram and trigram word clouds for top language

### Modeling
- Created and evaluated various classification models, varying the data preprocessing methods and associated hyperparameters.
