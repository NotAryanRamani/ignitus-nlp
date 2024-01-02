import pandas as pd
import numpy as np

from sklearn.linear_model  import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import pickle
import os
import warnings
warnings.filterwarnings("ignore")


class ModelTrainer:
    def __init__(self, tf_idf_features=2000):
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf_vectorizer = TfidfVectorizer(max_features = tf_idf_features)
        self.scaler = StandardScaler(with_mean=False)
        self.artifacts_folder = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)),'artifacts')
    
    def load_data(self, file):
        data = pd.read_csv(f'{self.artifacts_folder}/data/{file}')
        return data

    
    def clean_data(self, texts):
        print('Cleaning Texts')
        cleaned_texts = []
        stopwords_eng = stopwords.words('english')
        count = 0
        for text in texts:
            #removing hyperlinks
            text_ = re.sub(r'https?://[^\s\n\r]+', ' ', text)
            #removing punctuations and digits, only taking words
            text_ = re.sub('[^a-zA-z]', ' ', text_)
            text_ = re.sub(r'[\W_]+', ' ', text_)
            text_ = text_.lower()
            words = nltk.word_tokenize(text_)
            words = [word for word in words if word not in stopwords_eng]
            cleaned_texts.append(' '.join(words))
            count += 1
            if count % 50000 == 0:
                print(f'Cleaned {count} text')
        print(f'Cleaned {count} text')
        print('Completed cleaning text')
        return np.array(cleaned_texts)


    def lemmatize_text(self, texts):
        print('Lemmatizing Texts')
        lem_texts = []
        count = 0
        for text in texts:
            lem_words = [self.lemmatizer.lemmatize(word) for word in nltk.word_tokenize(text)]
            lem_text = ' '.join(lem_words)
            lem_texts.append(lem_text)
            count += 1
            if count % 50000 == 0:
                print(f'Lemmatized {count} text')
        print(f'Lemmatized {count} text')
        return np.array(lem_texts)


    def train_models(self, X_train, y_train):
        logistic = LogisticRegression(max_iter = 100, multi_class = 'multinomial', C = 0.001)
        multiNB  = MultinomialNB()
        
        X_train_vectorized = self.tfidf_vectorizer.fit_transform(X_train)
        self.scaler.fit(X_train_vectorized)
        X_train_scaled = self.scaler.transform(X_train_vectorized)
        print('Training Models')
        logistic.fit(X_train_scaled, y_train)
        multiNB.fit(X_train_vectorized, y_train)
        print('Models Trained')
        return (logistic, multiNB)

    def test_models(self, models, X_test, y_test):
        X_test_vectorized = self.tfidf_vectorizer.transform(X_test).toarray()
        X_test_scaled = self.scaler.transform(X_test_vectorized)

        logistic, multiNB = models
        
        test_score = logistic.score(X_test_scaled, y_test)
        print(f'Test Score for {logistic.__class__.__name__}: {test_score}')
        y_pred = logistic.predict(X_test_scaled)
        f1_score_ = f1_score(y_test, y_pred)
        print(f'F1 score for {logistic.__class__.__name__} = {f1_score_}')

        test_score = multiNB.score(X_test_vectorized, y_test)
        print(f'Test Score for {multiNB.__class__.__name__}: {test_score}')
        y_pred = multiNB.predict(X_test_scaled)
        f1_score_ = f1_score(y_test, y_pred)
        print(f'F1 score for {multiNB.__class__.__name__} = {f1_score_}')

    def save_models(self, models):
        print('Saving Models')
        os.makedirs(f'{self.artifacts_folder}/models', exist_ok=True)
        for model in models:
            with open(f'{self.artifacts_folder}/models/{model.__class__.__name__}.pkl', 'wb') as f:
                pickle.dump(model, f)
        print('Models Saved')
        
        