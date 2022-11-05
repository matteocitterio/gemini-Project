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
from datetime import datetime, timedelta

def to_celsius(temp):

    """
    Converts ferenheit to celsius
    """

    return (temp - 32) * 5/9

# Loading dataset
def load_data(path_name):
    """
    This function loads data into a pd.DataFrame and stripes measure units. Additionally,
    it converts into [0,1] numbers and drops some columns.
    #Params:
        -) `path_name` : `str` - location of the file
    #Returns: 
        -) `df` : `Pandas.DataFrame` - the loaded and striped data
        -) `max_temp` : `double` - maximum observed temperature used for data normalization
    """

    df = pd.read_csv(path_name)

    # this stripes out all the measure units
    head_list = [
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

    df.drop(columns=['Wind'], axis=1, inplace=True)
    df.dropna(how='any', inplace=True)
    max_temp = 0

# Deleating units in dataset using str.plit function

    for header in head_list:
        df[header] = (df[header].str.split().str[0])
        if (header == 'Temperature'):
            max_temp = df[header].max()
        df[header] = (df[header].astype(float))
        df[header] = df[header]/df[header].max()

    return df, max_temp

def CorrMatrix(data):
    """
    Plots an `sns.heatmap` which represents the correlation matrix
    #Params:
        -)`data` : `Pandas.DataFrame` - the data used for building the correlation matrix
    """

    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(10, 8))
    ax = sns.heatmap(data.corr(), vmin=-1, vmax=1,
                     cmap="coolwarm",
                     ax=ax,
                     annot=True)
    plt.show()

def GetLastWeekData(df):
    """
    This function returns a new dataframe containing only data related to the last week (the lastest 7 days)
    #Params:
        -) `df` : `Pandas.Dataframe` - The data you want to reduce
    #Returns: 
        -) `ReducedData` : `Pandas.DataFrame` - The reduced dataframe
    """

    LastAvailableDay = datetime.strptime(
        df['Time'].str.split().str[2].unique()[-1], "%Y-%m-%d")
    StartingDay = LastAvailableDay - timedelta(days=7-1)
    df['Time'] = df['Time'].str.split().str[2]
    
    ReducedData = df[df['Time'] >= StartingDay.strftime("%Y-%m-%d")]

    return ReducedData

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
