from flask import Flask, Response, request
from celery import Celery
from deep_learn.att_sentiment_classifier import ATTSentimentClassifier
from deep_learn.predict_sentiment import PredictSentiment
import pickle
from keras.models import load_model
import os

app = Flask(__name__)

app.config['CELERY_BROKER_URL'] = 'pyamqp://guest@localhost//'
app.config['CELERY_RESULT_BACKEND'] = 'rpc://'

celery = Celery(app.name, backend=app.config['CELERY_RESULT_BACKEND'],
             broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

@app.route('/')
def about_application():
    return 'An application for data science and around it...'

@celery.task
def train_att_sentiment_classifier():
    att_sentiment_classifier = ATTSentimentClassifier()
    att_sentiment_classifier._train_model()

if not os.path.isfile("reviews_tokenizer.pkl"):
    train_att_sentiment_classifier.apply_async()

with open('reviews_tokenizer.pkl', 'rb') as f:
    reviews_tokenizer = pickle.load(f)

reviews_classifier = load_model("deeplearn_sentiment_model.h5")

@app.route('/traindeeplearn')
def train_deep_learn():

    train_att_sentiment_classifier.apply_async()
    return Response(
        mimetype='application/json',
        status=200
    )

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    review = request.args.get('review')
    prediction = PredictSentiment()
    predicted = prediction.predict(review, reviews_tokenizer, reviews_classifier)
    return Response(
        mimetype='application/json',
        status=200,
        response=predicted
    )

if __name__ == '__main__':
    app.run()
