import pandas as pd
from scikitlearn_classifiers.constants import *
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import itertools
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from string import punctuation
from nltk.corpus import stopwords
import asyncio
import spacy
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

class TextCategorizer:

    def __init__(self):
        self.__spacyNlp = spacy.load("en")

    @classmethod
    async def clean_data(cls, text : str, remove_stops : bool = False, stemming : bool = False, lemmatization : bool = False) -> str:

        txt = ' '.join([w for w in text.split() if len(w) > 1])

        # Remove urls and emails
        txt = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', txt, flags=re.MULTILINE)
        txt = re.sub(r'[\w\.-]+@[\w\.-]+', ' ', txt, flags=re.MULTILINE)

        # Remove punctuation from text
        txt = txt.translate(punctuation)

        # Remove all symbols
        txt = re.sub(r'[^A-Za-z0-9\s]', r' ', txt)
        txt = re.sub(r'\n', r' ', txt)
        txt = re.sub(r'\s+', r' ', txt)

        # Replace words like sooooooo with so
        txt = ''.join(''.join(s)[:2] for _, s in itertools.groupby(txt))

        if remove_stops:
            stops = ['the', 'a', 'an', 'and', 'but', 'if', 'or', 'because', 'as', 'what', 'which', 'this', 'that', 'these',
                     'those', 'then',
                     'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during', 'to',
                     'what', 'which',
                     'is', 'if', 'while', 'this']
            txt = " ".join([w for w in txt.split() if w not in stops])
        if stemming:
            st = PorterStemmer()
            txt = " ".join([st.stem(w) for w in txt.split()])

        if lemmatization:
            wordnet_lemmatizer = WordNetLemmatizer()
            txt = " ".join([wordnet_lemmatizer.lemmatize(w, pos='v') for w in txt.split()])

        return txt

    def _clean_texts(self, texts : pd.Series) -> pd.Series:
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        tasks = ()
        tasks = (self.clean_data(text, True, False, True) for text in texts)
        try:
            result = event_loop.run_until_complete(asyncio.gather(*tasks))
        finally:
            event_loop.close()
        return result

    def __get_ner_tagged_text(self, text : str) -> str:
        tokens = self.__spacyNlp(text)

        for ent in tokens.ents:
            if ent.label_ in ("DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"):
                text = text.replace(ent.text, ent.label_)
        return text

    def ner_tagging_replacement(self, texts) -> list:
        tagged_txts= [self.__get_ner_tagged_text(text) for text in texts]
        return tagged_txts

    def train_categorizer(self, train_data : pd.DataFrame) -> None:
        train_data.text = category_classifier._clean_texts(train_data.text)
        train_data.text = category_classifier.ner_tagging_replacement(train_data.text)
        train_data['category_id'] = train_data['category'].factorize()[0]
        category_id_df = train_data[['category', 'category_id']].drop_duplicates().sort_values('category_id')
        category_to_id = dict(category_id_df.values)
        id_to_category = dict(category_id_df[['category_id', 'category']].values)

        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                                stop_words=stopwords.words('english'))
        features = tfidf.fit_transform(train_data.text).toarray()
        labels = train_data.category_id
        print(features.shape)

        N = 2
        for category, category_id in sorted(category_to_id.items()):
            features_chi2 = chi2(features, labels == category_id)
            indices = np.argsort(features_chi2[0])
            feature_names = np.array(tfidf.get_feature_names())[indices]
            unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
            bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
            print("# '{}':".format(category))
            print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
            print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

        models = [
            PassiveAggressiveClassifier(random_state=0, n_jobs=-1),
            ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=0, verbose=1),
            RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0),
            LinearSVC(random_state=0, verbose=1),
            MultinomialNB(),
            LogisticRegression(n_jobs=-1, random_state=0, verbose=1),
            KNeighborsClassifier(n_jobs=-1, n_neighbors=20)
        ]
        CV = 5
        cv_df = pd.DataFrame(index=range(CV * len(models)))
        entries = []
        for model in models:
            model_name = model.__class__.__name__
            accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
            for fold_idx, accuracy in enumerate(accuracies):
                entries.append((model_name, fold_idx, accuracy))
        cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

        plt.figure(figsize=(14, 6))
        sns.boxplot(x='model_name', y='accuracy', data=cv_df)
        sns.stripplot(x='model_name', y='accuracy', data=cv_df,
                      size=8, jitter=True, edgecolor="gray", linewidth=2)
        plt.show()

        mean_acc = cv_df.groupby('model_name').accuracy.mean()
        max_idx = pd.Series(mean_acc.values).idxmax()
        training_model = models[max_idx]

        training_model.fit(features, labels)
        pickle.dump(training_model, open("../category_classifier.pkl", "wb"))

if __name__=='__main__':
    train_data = pd.read_csv(MULTICATEGORY_TRAIN_FILE)
    category_classifier = TextCategorizer()
    category_classifier.train_categorizer(train_data)