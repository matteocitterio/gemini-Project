#!/usr/bin/env python
# -*- coding: utf-8 -*- #

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from plot_keras_history import show_history, plot_history
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout, GRU, Bidirectional, Conv1D, MaxPooling1D
from hyperopt import hp, tpe, Trials, fmin
from tensorflow.keras.layers import Dense, Flatten
from hyperopt import fmin, tpe, hp, Trials, space_eval, STATUS_OK
import time
import sklearn
import seaborn as sns
import os

def to_celsius(temp):

    """
    Converts ferenheit to celsius
    """

    return (temp - 32) * 5/9


def load_data(path_name):
    """
    This function loads data into a pd.DataFrame and stripes measure units. Additionally,
    it converts into [0,1] numbers and drops some columns
    - path_name: str location of the file
    return: Pandas.DataFrame
    """
    df=pd.read_csv(path_name)

    #this stripes out all the measure units
    head_list=[
        'Temperature',
        'Humidity',
        'Dew Point',
        'Speed',
        'Gust',
        'Pressure',
        'Precip. Rate.',
        'Precip. Accum.',
        'Solar',
    ]

    df.dropna(how='any', inplace=True)
    df.drop(columns=['Wind'], axis=1, inplace=True)

    for header in head_list:
        df[header] =  (df[header].str.split().str[0])
        df[header] =  (df[header].astype(float))
        df[header]=df[header]/df[header].max()

    
    df.dropna(how='any', inplace=True)

    # df.rename(columns={
    #     'Temperature': 'Temperature[°C]',
    #     'Dew Point': 'Dew Point[°C]',
    #     'Humidity': 'Humidity[%]',
    #     'Speed': 'Speed[kmh]',
    #     'Gust': 'Gust[kmh]',
    #     'Pressure': 'Pressure[in]',
    #     'Precip. Rate.': 'Precip. Rate. [in/hr]',
    #     'Precip. Accum.': 'Precip. Accum.[in]',
    #     'Solar': 'Solar [w/m2]'
    #     }, inplace=True)

    return df

def LSTM_model_tot(input_shape,activation='tanh',dropout=0.2):

    """
    Builds an LSTM.keras.model with encoder(?)
    #params:
    -) activation [str]: activation function, default is tanh
    -) input_shape [tuple]: input shape
    -) droput [double]: dropout parameter, default is 0.2

    return: it returns the model
    """

    enco_deco=tf.keras.models.Sequential()
    enco_deco.add(Bidirectional(LSTM(75, activation=activation, return_sequences=True), input_shape=input_shape))
    enco_deco.add(Dropout(dropout))   
    enco_deco.add((Dense(units=1)))

    return enco_deco