# Text Classification
## This repository allows usage of text classification models as a service(using flask and celery).
### Trained on IMDB data

#### Deep learn model is derived from:
##### 1. [Text classifier by richliao](https://github.com/richliao/textClassifier)
##### 2. [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)

##### Fetch training data from: https://www.kaggle.com/c/word2vec-nlp-tutorial/data

##### Fetch contractions model from: https://code.google.com/archive/p/word2vec/ or https://github.com/mmihaltz/word2vec-GoogleNews-vectors

##### Internally here, sentiment classification model uses Keras deep learn library which is based on tensorflow. 

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

python3 -m pip install --user virtualenv
python3 -m virtualenv env
source env/bin/activate

pip install -r requirements.txt

# update constants for model
nano deeplearn/constants.py

./venv/bin/python ./deeplearn/att_sentiment_classifier.py

# Run Flask service
./venv/bin/python -m flask run
