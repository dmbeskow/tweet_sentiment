# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 22:55:39 2019

This contains an LSTM model for  predicting sentiment

@author: David Beskow
"""

from keras.models import load_model
import keras
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text
import os, gzip, json, io
import pandas as pd
import pkg_resources
    

#%%


def sentiment_prob(text_list):
    '''
    This function predicts sentiment of a list of text and returns a probability.
    
    Probabilites are between 0 and 1, with predictions closer to 0 being more negative,
    and predictions closer 1 are more positive.  
    '''

    model_path = pkg_resources.resource_filename('tweet_sentiment', 'data/sentiment_LSTM_20190320.h5')
    lstm_model = load_model(model_path)
    
    tokenizer_path = pkg_resources.resource_filename('tweet_sentiment', 'data/sentiment_tokenizer.json.gz')
    with io.TextIOWrapper(gzip.open(tokenizer_path, 'r')) as f:
        data = json.load(f)
    tokenizer = keras.preprocessing.text.tokenizer_from_json(data)

    new_token = tokenizer.texts_to_sequences(text_list)
    new_token = sequence.pad_sequences(new_token, maxlen=116)

    yhat = lstm_model.predict(new_token)
    yhat = yhat.T[0].tolist()
    
    return(yhat)
