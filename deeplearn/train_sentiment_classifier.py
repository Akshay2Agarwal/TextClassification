import re
import numpy as np
from bs4 import BeautifulSoup
import asyncio
from nltk import tokenize
import gensim
from deeplearn import fetch_data, constants, attention_layer
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding, Dense, Input, Dropout, LSTM, Bidirectional, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from pycontractions import Contractions
import tensorflow as tf
import pickle


class ATTSentimentClassifier(object):

    @classmethod
    async def clean_str(cls, strings):
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
    def clean_reviews(cls, reviews):
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        tasks = (cls.clean_str(review) for review in reviews)
        try:
            result = event_loop.run_until_complete(asyncio.gather(*tasks))
        finally:
            event_loop.close()
        return result

    @classmethod
    def _tokenize_sentences(cls, reviews):
        for review in reviews:
            yield tokenize.sent_tokenize(review)

    @classmethod
    def train_model(cls):
        dao = fetch_data.FetchData()
        train_data = dao._fetch_imdb_train_data()

        cont = Contractions(constants.CONTRACTIONS_BIN_FILE)
        cont.load_models()

        for index, row in train_data.iterrows():
            row.review = BeautifulSoup(row.review, features="html.parser").get_text()

        train_data.review = cls.clean_reviews(train_data.review)

        reviews = list(cls._tokenize_sentences(train_data.review))

        labels = list(train_data.sentiment)

        tokenizer = Tokenizer(num_words=constants.MAX_NB_WORDS)
        tokenizer.fit_on_texts(train_data.review)

        data = np.zeros((len(train_data.review), constants.MAX_SENTS, constants.MAX_SENT_LENGTH), dtype='int32')

        words = list()
        for i, sentences in enumerate(reviews):
            for j, sent in enumerate(sentences):
                if j < constants.MAX_SENTS:
                    wordTokens = text_to_word_sequence(sent)
                    k = 0
                    for _, word in enumerate(wordTokens):
                        if k < constants.MAX_SENT_LENGTH and tokenizer.word_index[word] < constants.MAX_NB_WORDS:
                            data[i, j, k] = tokenizer.word_index[word]
                            k = k + 1
                    words.append(wordTokens)

        word_index = tokenizer.word_index
        print('Total %s unique tokens.' % len(word_index))

        labels = to_categorical(np.asarray(labels))
        print('Shape of data tensor:', data.shape)
        print('Shape of label tensor:', labels.shape)

        wordSkipGramModel = gensim.models.Word2Vec(words, min_count=5, size=constants.EMBEDDING_DIM, window=4, sg=1)

        word_embedding_matrix = np.random.random((len(word_index) + 1, constants.EMBEDDING_DIM))
        for word, i in word_index.items():
            try:
                word_embedding_vector = wordSkipGramModel.wv.get_vector(word)
            except KeyError:
                continue
                # words not found in embedding index will be all-zeros.EMBEDDING_DIM
            if word_embedding_vector is not None:
                word_embedding_matrix[i] = word_embedding_vector

        embedding_layer = Embedding(len(word_index) + 1, constants.EMBEDDING_DIM, weights=[word_embedding_matrix],
                                    input_length=constants.MAX_SENT_LENGTH, trainable=True)

        sentence_input = Input(shape=(constants.MAX_SENT_LENGTH,), dtype='float64')
        embedded_sequences = embedding_layer(sentence_input)
        sentence_lstm = Bidirectional(LSTM(200, return_sequences=True))(embedded_sequences)
        l_dropout = Dropout(0.5)(sentence_lstm)
        l_dense = TimeDistributed(Dense(400))(l_dropout)
        l_att = attention_layer.AttLayer()(l_dense)
        l_dropout_1 = Dropout(0.4)(l_att)
        sentEncoder = Model(sentence_input, l_dropout_1)

        review_input = Input(shape=(constants.MAX_SENTS, constants.MAX_SENT_LENGTH), dtype='float64')
        review_encoder = TimeDistributed(sentEncoder)(review_input)
        review_dropout = Dropout(0.3)(review_encoder)
        l_lstm_review = Bidirectional(LSTM(100, return_sequences=True))(review_dropout)
        l_att_dropout_review = Dropout(0.2)(l_lstm_review)
        l_dense_review = TimeDistributed(Dense(200))(l_att_dropout_review)
        l_dropout_review = Dropout(0.2)(l_dense_review)
        l_att_review = attention_layer.AttLayer()(l_dropout_review)
        preds = Dense(2, activation='softmax')(l_att_review)
        model = Model(review_input, preds)
        adam = Adam(lr=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(data, labels, validation_split=0.2, epochs=10, batch_size=50, shuffle=False, verbose=1)
        model.save('deeplearn_sentiment_model.h5')

        # Save Tokenizer i.e. Vocabulary
        with open('reviews_tokenizer.pkl', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    ATTSentimentClassifier().train_model()
