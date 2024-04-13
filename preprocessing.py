import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

class Preprocessing:
    def __init__(self):
        self.cv = CountVectorizer()
        self.le = LabelEncoder()

    def clean_text(self, text):
        nonalphanumeric = ['\'', '.', ',', '\"', ':', ';', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '+', '=', '[', ']', '{', '}', '\\', '?', '/', '>', '<', '|', ' ']
        stopwords_list = nonalphanumeric
        tokens = word_tokenize(text) 
        words = [word.lower() for word in tokens if word not in stopwords_list]
        words = [PorterStemmer().stem(word) for word in words] 
        return " ".join(words) 

    def preprocess_data(self, data_path):
        data = pd.read_csv(data_path, encoding='utf-8').copy()
        data['clean_text'] = data['Text'].apply(self.clean_text)
        data['language_encoded'] = self.le.fit_transform(data['language'])
        x = self.cv.fit_transform(data['clean_text'])
        x = x.astype('uint8')
        y = data['language_encoded']
        return x, y, data
