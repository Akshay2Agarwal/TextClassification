# Text Classification
## This repository allows usage of text classification models as a service.
### Trained on IMDB data

#### Deep learn model is derived from:
##### 1. [Text classifier by richliao](https://github.com/richliao/textClassifier)
##### 2. [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)

##### Fetch training data from: https://www.kaggle.com/c/word2vec-nlp-tutorial/data

##### Fetch contractions model from: https://code.google.com/archive/p/word2vec/ or https://github.com/mmihaltz/word2vec-GoogleNews-vectors

To use the same:
```
# prerequisites
Rabbitmq
Python 3.5
Flask

# clone the repository
git clone {repo address}

# install Dependent library
cd TextClassification
pip install -r requirements.txt

# update constants for model
nano deeplearn/constants.py