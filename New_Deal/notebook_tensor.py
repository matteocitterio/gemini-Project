#!/usr/bin/env python
# -*- coding: utf-8 -*- #

"""
Import all the basic stuff
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

"""
For Ferenheit conversion
"""

from gemlib import to_celsius

import IPython
import IPython.display
import matplotlib as mpl

"""
import tensorflow stuff
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.backend import square, mean
from plot_keras_history import show_history, plot_history

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

df.Wind = df.Wind.replace(['North'], 'N')
df.Wind = df.Wind.replace(['South'], 'S')
df.Wind = df.Wind.replace(['East'], 'E')
df.Wind = df.Wind.replace(['West'], 'W')

wind_dirs=['N', 'NNE', 'NNW', 'NW', 'NE', 'ENE', 'E', 'SE', 'ESE','SSE', 'WNW', 'W', 'S', 'SW', 'SSW', 'WSW']
angle_list=[0,22.5,337.5,315,45,67.5,90,135,112.5,157.5,295.5,270,180,225,202.5,247.5]


def direction_to_angle(direction, wind_dirs, angle_list):
    list_wind = []
    for i in range(0, len(direction)):
        index = wind_dirs.index(direction.iloc[i])
        list_wind.append(angle_list[index])
    return list_wind

df.Wind=direction_to_angle(df.Wind,wind_dirs,angle_list)

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

# plt.hist2d(df['Wx'], df['Wy'], bins=(50, 50), vmax=400)
# plt.colorbar()
# plt.xlabel('Wind X [m/s]')
# plt.ylabel('Wind Y [m/s]')
# ax = plt.gca()
# ax.axis('tight')

# plt.show()

"""
Let's cut the complete dataset eliminating the parts where no data are available, namely consider only from
30-10-2021 until today
"""

LastAvailableDay = datetime.strptime(
        df['Time'].str.split().str[2].unique()[-1], "%Y-%m-%d")
StartingDay = LastAvailableDay - timedelta(days=375-1)
print(StartingDay)

df['Temperature'] = to_celsius(df['Temperature'])

"""
We can now think that time is of course periodic and an a priori assumption could be considering weather as
something with the same period of the time of the Year + the time of the day + fluctuations.
We should then convert the time info into a periodic variable
"""

date_time = pd.to_datetime(df.pop('Time'), format='%I:%M %p %Y-%m-%d')
timestamp_s=date_time.map(pd.Timestamp.timestamp)

day = 24*60*60
year = (365.2425)*day
hour = 60 * 60

df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
df['Hour'] = (timestamp_s - min(timestamp_s)) / hour

# fig2, ax2 = plt.subplots(figsize=(10, 8))
# ax2.plot(np.array(df['Hour'])[:576],np.array(df['Day sin'])[:576])
# ax2.plot(np.array(df['Hour'])[:576],np.array(df['Day cos'])[:576])
# ax2.set_xlabel('Time [h]')
# ax2.set_title('Time of day signal')

# fig3, ax3 = plt.subplots(figsize=(10, 8))
# ax3 = sns.heatmap(df.corr(), vmin=-1, vmax=1, cmap='coolwarm', annot=True)

"""
Precipitation rate has only spikes and it is not as informative since it basically doesnt correlate with anything
we should drop it and keep in mind that the same fate could be shared by her sister, `Prep. Acc.`
"""

df.pop('Precip. Rate.')
df.head(5)

"""
Since i need to test what i'im doing and to stress my architecture features I only consider the 2% of the
complete dataset which is something around a week
"""

n = len(df)
df = df[int(n*0.8):-1]

"""
Let's get the features and the data split: 70% training, 20% validation, 10% test
"""

column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

Hours = df['Hour']
Days = Hours/24

fig4, ax4 = plt.subplots(figsize=(10, 8))
ax4.plot(df['Hour']/24, df['Temperature'])
ax4.set_xlabel('Days')
ax4.set_ylabel('Temperature[°C]')
ax4.axvspan(xmin=train_df['Hour'].iloc[0]/24,
            xmax=train_df['Hour'].iloc[-1]/24,alpha=0.1, color='red', label='train')
ax4.axvspan(xmin=val_df['Hour'].iloc[0]/24,
            xmax=val_df['Hour'].iloc[-1]/24, alpha=0.1, color='green', label='val')
ax4.axvspan(xmin=test_df['Hour'].iloc[0]/24,
            xmax=test_df['Hour'].iloc[-1]/24, alpha=0.1, color='blue', label='test')
ax4.legend()
ax4.set_title('Temperature Dataset Split')


train_df.pop('Hour')
val_df.pop('Hour')
test_df.pop('Hour')
df.pop('Hour')

plt.show()

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

class WindowGenerator():

    """
    The constructor of this class takes in input the window width, the shift and the data and produces as
    an output the right set of indexis for a desired window
    """

    def __init__(self, input_width, label_width, shift,
                 train_df=train_df, val_df=val_df, test_df=test_df,
                 label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]

        """
        Let's make a simple example. Consider that your data are collected hourly so that you have a point of
        your series every hour. Let's say you want to consider as `input` of your model a sequence of 24 hours 
        and you want to predict the next two hours, so that you have as label a sequence of 2 points. In total
        (if you don't want your input sequence to be overlapped by labels, put shift=inputs_len) you'd have some-
        thing like:

        --Python:

        `INPUT_STEPS = 24
        OUT_STEPS = 2 # = len(labels)
        Window_example = WindowGenerator(input_width=INPUT_STEPS,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS)`

        --Return:

        `Total window size: 26
        Input indices: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
        Label indices: [24 25]
        Label column name(s): None`

        Where, if not specified, the  default Label column name(s) is 'Temperature'

        """

    def __repr__(self):

        """
        This simply prints out what follows whenever a new window is built
        """

        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):

        """
        This effectively splits the data (with all the related features) into the desired window shape (inputs,
        labels)
        """

        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]]
                    for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        """ 
        The `None` shape simply means that it could be anything, it is not fixed.
        """
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col='Temperature', max_subplots=3):

        """
        Useful function that plot the different quantities on an example window of data. The example
        retrieved throgh `self.example` method is random.
        """

        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(
                    plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')

    def make_dataset(self, data):

        """
        This function makes a keras dataset diveded into batches.
        """

        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)

        ds = ds.map(self.split_window)

        return ds

    """
    So if you make:

    -- Python:

    `print(Window_example.train.element_spec)`

    -- Return:

    `(TensorSpec(shape=(None, 24, 14), dtype=tf.float32, name=None),
    TensorSpec(shape=(None, 2, 14), dtype=tf.float32, name=None))`

    Where the first tensor has shape (batch, time steps, features) and the second (batch, time steps, label cols)
    where the label cols size is =14 (=features) simply because I did'nt specified a label_col for the moment.

    By iterating over a `Dataset` yields to concrete batches:

    --Python:

    `for example_inputs, example_labels in Window_example.train.take(1):
        print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
        print(f'Labels shape (batch, time, features): {example_labels.shape}')
    `

    --Returns:

    `Inputs shape (batch, time, features): (32, 24, 14)
    Labels shape (batch, time, features): (32, 2, 14)`

    Where  `.take(1)` extracts a single batch.

    """

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result

        return result


MAX_EPOCHS = 20


def compile_and_fit(model, window, patience=5):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history

input_hours=24
output_hours=1

OUT_STEPS = 12 * output_hours
IN_STEPS = 12 * input_hours
prediction_labels=['Temperature']
num_features_predicted = len(prediction_labels)
multi_window = WindowGenerator(input_width=IN_STEPS,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS,
                               label_columns=prediction_labels)

multi_window.plot()

multi_val_performance = {}
multi_performance = {}

multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units].
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(32, return_sequences=False),
    # Shape => [batch, out_steps*features].
    tf.keras.layers.Dense(OUT_STEPS*num_features_predicted,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features].
    tf.keras.layers.Reshape([OUT_STEPS, num_features_predicted])
])

history = compile_and_fit(multi_lstm_model, multi_window)

multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
multi_performance['LSTM'] = multi_lstm_model.evaluate(
    multi_window.test, verbose=0)

plot_history(history)

def plot_renormalized(self, model=None, plot_col='Temperature', max_subplots=3):
    a = list(np.arange(5, 65, 5))
    str_list = ['+'+str(i) for i in a]
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [°C]')
            # plt.plot(self.input_indices, (inputs[n, :, plot_col_index]+train_mean)*train_std)
            # label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(
                    plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.plot(self.label_indices, ((
                labels[n, :, label_col_index]*train_std[0])+train_mean[0]), '-', c='#2ca02c')
            plt.scatter(self.label_indices, ((
                labels[n, :, label_col_index]*train_std[0])+train_mean[0]), edgecolors='k', label='Labels', c='#2ca02c', s=64)

            if model is not None:
                predictions = model(inputs)

            plt.plot(self.label_indices, ((
                predictions[n, :, label_col_index]*train_std[0])+train_mean[0]), '-', c='#ff7f0e', ms=1)
            plt.scatter(self.label_indices, ((predictions[n, :, label_col_index]*train_std[0])+train_mean[0]), marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

            plt.xticks(ticks=list(np.arange(288, 288+12)), labels=str_list)

    plt.xlabel('minutes into the future')

WindowGenerator.plot_renormalized = plot_renormalized

multi_window.plot_renormalized(multi_lstm_model)
plt.show()

enco_deco_model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, return_sequences=False)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(OUT_STEPS*num_features_predicted,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features].
    tf.keras.layers.Reshape([OUT_STEPS, num_features_predicted])

])
history = compile_and_fit(enco_deco_model, multi_window)

multi_val_performance['enco_deco'] = enco_deco_model.evaluate(multi_window.val)
multi_performance['enco_deco'] = enco_deco_model.evaluate(
    multi_window.test, verbose=0)

plot_history(history)
multi_window.plot_renormalized(enco_deco_model)

plt.show()

# bidirectional_spinto_model = tf.keras.models.Sequential([

#     tf.keras.layers.Bidirectional(
#         tf.keras.layers.LSTM(1024, return_sequences=True)),
#     tf.keras.layers.Bidirectional(
#         tf.keras.layers.LSTM(512, return_sequences=True)),
#     tf.keras.layers.Bidirectional(
#         tf.keras.layers.LSTM(256, return_sequences=True)),
#     tf.keras.layers.Bidirectional(
#         tf.keras.layers.LSTM(128, return_sequences=True)),
#     tf.keras.layers.Bidirectional(
#         tf.keras.layers.LSTM(64, return_sequences=False)),
#     tf.keras.layers.Dense(OUT_STEPS),
#     tf.keras.layers.Reshape([OUT_STEPS, num_features_predicted])
# ])

# history = compile_and_fit(bidirectional_spinto_model, multi_window)

# multi_val_performance['bidirectional_spinto_model'] = bidirectional_spinto_model.evaluate(
#     multi_window.val)
# multi_performance['bidirectional_spinto_model'] = bidirectional_spinto_model.evaluate(
#     multi_window.test, verbose=0)
# multi_window.plot_renormalized(bidirectional_spinto_model)
# plot_history(history)

x = np.arange(len(multi_performance))
width = 0.3

print(multi_performance)
print(multi_val_performance)
print(multi_lstm_model.metrics_names)
print(multi_lstm_model.metrics_names.index('mean_absolute_error'))
print(v[metric_index] for v in multi_val_performance.values())
print(v[metric_index] for v in multi_performance.values())

metric_name = 'mean_absolute_error'
metric_index = multi_lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in multi_val_performance.values()]
test_mae = [v[metric_index] for v in multi_performance.values()]

plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=multi_performance.keys(),
           rotation=45)
plt.ylabel(f'MAE (average over all times and outputs)')
_ = plt.legend()
plt.show()




