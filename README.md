## Project Goal
The goal of this project is to build a model that can predict the main programming language of from README file of Github repository.
This project started by scraping README files from various GitHub repositories using web scraping techniques. Following the acquisition and preparation of our data, our team used natural language processing exploration methods such as word clouds, bigrams, and trigrams. We used multiclass classification methods to create multiple machine learning models.
The end goal was to create an NLP model that accurately predicted the programming language used in a github repository based on the words and word combinations found in the readme files.


## Project Outline:
### Acquisiton of data:
- Conduct web scraping of repositories' readme contents.
- 
### Prepare and clean data
- Drop nulls
- Lowercase all text
- Remove stopwords (including customer, customers, 1, 2, and i)
- Tokenize the data
- Stem and Lemmatize


### Exploration

- bigrams and trigrams for top language
- Analyze bigram and trigram word clouds for top language

### Modeling
