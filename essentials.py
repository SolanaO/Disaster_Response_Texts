import os
import re
import json
import pickle
import joblib

import numpy as np
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
#from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify


from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)

# this tokenizer provided by udacity
def tokenize(text):

    """
    Takes raw text, removes punctuation signs, substitutes
    with spaces, normalizes to lowercase, tokenizes text, lemmatizes, and returns list of clean tokens.

    INPUT:
        raw message (str)
    OUTPUT:
        clean tokens (list)

    """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def tokenize_myvers(text):
    """
    Contains the pre-processing steps for a message:
    - replaces url links with placeholders
    - removes punctuation, unusual characters
    - tokenize,
    - lemmatize
    - lowercasing
    - removes stopwords in English language
    INPUT: string, raw message
    OUTPUT: list of clean tokens
    """
    # use regular expressions to detect an url
    url_string = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"

    # get the list of all urls, emails using regex
    detected_urls = re.findall(url_string, text)

    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    # remove punctuation and unusual characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text).strip()

    # split into words
    words = word_tokenize(text)

    # lemmatize - reduce words to their root form
    words = [WordNetLemmatizer().lemmatize(w) for w in words]

    # case normalize and remove leading & trailing empty spaces
    words = [w.lower().strip() for w in words]

    # remove stopwords
    clean_words = [w for w in words if w not in stopwords.words('english') or w in ['not', 'can']]

    return clean_words


# custom tokenizer class to use in the pipeline
class Tokenizer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def tokenize(text):
            """
            Contains the pre-processing steps for a message:
            - replaces url links with placeholders
            - removes punctuation, unusual characters
            - tokenize,
            - lemmatize
            - lowercasing
            - removes stopwords in English language
            INPUT: string, raw message
            OUTPUT: list of clean tokens
            """
            # use regular expressions to detect an url
            url_string = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"

            # get the list of all urls, emails using regex
            detected_urls = re.findall(url_string, text)

            # replace each url in text string with placeholder
            for url in detected_urls:
                text = text.replace(url, 'urlplaceholder')

            # remove punctuation and unusual characters
            text = re.sub(r"[^a-zA-Z0-9]", " ", text).strip()

            # split into words
            words = word_tokenize(text)

            # lemmatize - reduce words to their root form
            words = [WordNetLemmatizer().lemmatize(w) for w in words]

            # case normalize and remove leading & trailing empty spaces
            words = [w.lower().strip() for w in words]

            # remove stopwords
            clean_words = [w for w in words if w not in stopwords.words('english') or w in ['not', 'can']]

            return ' '.join(clean_words)
