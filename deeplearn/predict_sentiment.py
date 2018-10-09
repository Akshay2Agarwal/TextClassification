import re
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from keras.preprocessing.text import text_to_word_sequence
import numpy as np
from deeplearn import constants


class PredictSentiment():

    @classmethod
    def _clean_str(cls, strings):
        """
        Tokenization/string cleaning for dataset
        Every dataset is lower cased except
        """
        strings = str(strings)
        strings = re.sub(r"\\", "", strings)
        strings = re.sub(r"\'", "", strings)
        strings = re.sub(r"\"", "", strings)
        strings = re.sub(r"<br />", "", strings)
        return strings.strip().lower()

    @classmethod
    def predict(cls, review_to_predict, review_tokenizer, reviews_classifier):
        review_to_predict = BeautifulSoup(review_to_predict, features="html.parser").get_text()
        review_cleaned = cls._clean_str(review_to_predict)
        review_sents = sent_tokenize(review_cleaned)
        review_tokenized = np.zeros((1, constants.MAX_SENTS, constants.MAX_SENT_LENGTH), dtype='int64')

        for i, sentence in enumerate(review_sents):
            if i < constants.MAX_SENTS:
                wordTokens = text_to_word_sequence(sentence)
                k = 0
                for _, word in enumerate(wordTokens):
                    if k < constants.MAX_SENT_LENGTH and word in review_tokenizer.word_index and \
                            review_tokenizer.word_index[word] < constants.MAX_NB_WORDS:
                        review_tokenized[0, i, k] = review_tokenizer.word_index[word]
                        k = k + 1

        predictions = reviews_classifier.predict(review_tokenized)
        return predictions
