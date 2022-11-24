#!/usr/bin/env python
# -*- coding: utf-8 -*- #

"""
Import all the basic stuff
"""

from plot_keras_history import show_history, plot_history
from tensorflow.keras.backend import square, mean
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.keras.models import Sequential
import matplotlib as mpl
import IPython.display
import IPython
from predlib import direction_to_angle, WindowGenerator, Model, RepeatBaseline
from gemlib import to_celsius
from matplotlib.collections import TriMesh
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

"""
Automatically retrieves the path of the lastest `Bresso_updated` weather-data file downloaded through `table_scraper.py`
"""

gemini_path = '/Users/matteocitterio/Documents/gemini-Project'
files = [i for i in os.listdir(gemini_path) if os.path.isfile(os.path.join(gemini_path, i)) and
         'Bresso_updated' in i]

path_name = (files[-1])

"""
Read file and drop NaNs
"""

df = pd.read_csv(path_name)
df.dropna(how='any', inplace=True)

"""
Convert wind direction into degrees
"""

Written_winds = ['North', 'South', 'East', 'West']
Shorter_winds = ['N', 'S', 'E', 'W']

df.Wind = df.Wind.replace(Written_winds, Shorter_winds)

wind_dirs = ['N', 'NNE', 'NNW', 'NW', 'NE', 'ENE', 'E', 'SE',
             'ESE', 'SSE', 'WNW', 'W', 'S', 'SW', 'SSW', 'WSW']
angle_list = [0, 22.5, 337.5, 315, 45, 67.5, 90, 135,
              112.5, 157.5, 295.5, 270, 180, 225, 202.5, 247.5]

df.Wind = direction_to_angle(df.Wind, wind_dirs, angle_list)

"""
Stripe measure units and convert everything into float numbers
"""

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

for header in head_list:
    df[header] = (df[header].str.split().str[0])
    df[header] = (df[header].astype(float))

"""
Let's do the vector transformation (wind speed, wind direction) -> (wind x, wind y), this will improve our 
prediction
"""

wv = df.pop('Speed')

# Convert to radians.
wd_rad = df.pop('Wind')*np.pi / 180

# Calculate the wind x and y components.
df['Wx'] = wv*np.cos(wd_rad)
df['Wy'] = wv*np.sin(wd_rad)

"""
Let's cut the complete dataset eliminating the parts where no data are available, namely consider only from
30-10-2021 until today
"""

LastAvailableDay = datetime.strptime(
    df['Time'].str.split().str[2].unique()[-1], "%Y-%m-%d")
StartingDay = LastAvailableDay - timedelta(days=375-1)
print(StartingDay)

df = df[df['Time'].str.split().str[2] >= StartingDay.strftime("%Y-%m-%d")]

df['Temperature'] = to_celsius(df['Temperature'])

"""
We can now think that time is of course periodic and an a priori assumption could be considering weather as
something with the same period of the time of the Year + the time of the day + fluctuations.
We should then convert the time info into a periodic variable
"""

date_time = pd.to_datetime(df.pop('Time'), format='%I:%M %p %Y-%m-%d')
timestamp_s = date_time.map(pd.Timestamp.timestamp)

day = 24*60*60
year = (365.2425)*day
hour = 60 * 60

df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

"""
Precipitation rate has only spikes and it is not as informative since it basically doesnt correlate with anything
we should drop it and keep in mind that the same fate could be shared by her sister, `Prep. Acc.`
"""

df.pop('Precip. Rate.')

"""
Since i need to test what i'im doing and to stress my architecture features I only consider the 2% of the
complete dataset which is something around a week
"""

# n = len(df)
# df = df[int(n*0.95):-1]

"""
Let's get the features and the data split: 70% training, 20% validation, 10% test
"""

column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]
print('Num features:', num_features)

"""
Let's renormalize the dataset according to the train mean and std
"""

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std


"""
This is the core the class WindowGenerator which essentially carries out all the boring stuff we need to do in
order to create an eligible piece if training-val-test data.
"""


MAX_EPOCHS = 20

input_hours = 24
output_hours = 1

OUT_STEPS = 12 * output_hours
IN_STEPS = 12 * input_hours

prediction_labels = ['Temperature']
num_features_predicted = len(prediction_labels)
multi_window = WindowGenerator(input_width=IN_STEPS,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS,
                               train_df=train_df,
                               val_df=val_df,
                               test_df=test_df,
                               label_columns=prediction_labels,
                               batch_size=4)

# multi_window.plot()

inputs, labels=multi_window.example

# inputs, labels = multi_window.unshuffled_test.take(1)

x=[]
y_true=[]
y_pred_baseline=[]
y_pred_multi=[]
y_pred_lstm=[]
y_pred_dense=[]
y_pred_cnn=[]
y_pred_bidir=[]

model = Model(multi_window, train_std['Temperature']) 

# model.repeat_baseline('repeat_baseline')
# model.compile_and_fit()

# model.performance()
# predictions_baseline=model.predict(inputs)
# print(predictions_baseline.shape)
# predictions_baseline=tf.reshape(predictions_baseline, (4,12,1))

print('Multi')
model.multi_linear(OUT_STEPS, num_features_predicted, model_name='multilinear', LoadModel=True)

model.performance()
predictions = model.predict(inputs)

# print('lstm')
# model.multi_lstm(OUT_STEPS, num_features_predicted,
#                  model_name='multi_lstm', LoadModel=True)

# model.performance()
# predictions_lstm = model.predict(inputs)
print('Bidir')
# model.bidirectional_lstm(OUT_STEPS, num_features_predicted,
#                          model_name='bidirectional_lstm', LoadModel=True)

# model.performance()
# predictions_bidir = model.predict(inputs)

print('Dense')
model.dense(OUT_STEPS, num_features_predicted,
                 model_name='dense', LoadModel=True)

model.performance()
predictions_dense = model.predict(inputs)

# print('cnn')
model.cnn(OUT_STEPS, num_features_predicted,
                 model_name='cnn', LoadModel=True)

model.performance()
predictions_cnn = model.predict(inputs)

# model.comparison_performances()


model.comparison_performances()


for i in range(0, 4):

    for j in range(0, multi_window.label_width):

        x.append((multi_window.label_indices + (OUT_STEPS * i))[j])
        y_true.append(((labels[i, :, multi_window.label_columns_indices.get(
            'Temperature', None)]*train_std[0])+train_mean[0])[j])
        y_pred_multi.append(((predictions[i, :, multi_window.label_columns_indices.get(
            'Temperature', None)]*train_std[0])+train_mean[0])[j])
        # y_pred_lstm.append(((predictions_lstm[i, :, multi_window.label_columns_indices.get(
        #     'Temperature', None)]*train_std[0])+train_mean[0])[j])
        y_pred_dense.append(((predictions_dense[i, :, multi_window.label_columns_indices.get(
            'Temperature', None)]*train_std[0])+train_mean[0])[j])
        # y_pred_bidir.append(((predictions_bidir[i, :, multi_window.label_columns_indices.get(
        #     'Temperature', None)]*train_std[0])+train_mean[0])[j])
        y_pred_cnn.append(((predictions_cnn[i, :, multi_window.label_columns_indices.get(
            'Temperature', None)]*train_std[0])+train_mean[0])[j])
        # y_pred_baseline.append(((predictions_baseline[i, :, 0]*train_std[0])+train_mean[0])[j])


plt.plot(x, y_true, '-', lw=3, label='labels')
plt.plot(x, y_pred_multi, ls='--',label='multi linear')
# plt.plot(x, y_pred_lstm, '--', label='lstm')
plt.plot(x, y_pred_dense, ls='--', label='dense')
plt.plot(x, y_pred_cnn, ls='--', label='cnn')
# plt.plot(x, y_pred_baseline, '--', label='repeat baseline')
# plt.plot(x, y_pred_bidir, '--', label='bidir')
plt.legend()
plt.show()
