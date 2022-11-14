#!/usr/bin/env python
# -*- coding: utf-8 -*- #

from re import S
import numpy as np
import pandas as pd
from sklearn.base import OutlierMixin
import tensorflow as tf
import os
from datetime import datetime
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.backend import square, mean
from plot_keras_history import show_history, plot_history

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

def direction_to_angle(direction, wind_dirs, angle_list):
    """
    It takes a direction (str) and outputs a direction (float) in terms of degrees
    """
    list_wind = []
    for i in range(0, len(direction)):
        index = wind_dirs.index(direction.iloc[i])
        list_wind.append(angle_list[index])
    return list_wind


class WindowGenerator():

    """
    The constructor of this class takes in input the window width, the shift and the data and produces as
    an output the right set of indexis for a desired window
    """

    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None,batch_size=32):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size=batch_size

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
        plt.show()

    def make_dataset(self, data, shuffle=True, stride=1):
        """
        This function makes a keras dataset diveded into batches.
        """

        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=stride,
            shuffle=shuffle,
            batch_size=8,)

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
    def unshuffled_test(self):
        return self.make_dataset(self.test_df, shuffle=False, stride=self.label_width)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = None
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            randomator = tf.random.uniform(
                shape=[], maxval=self.batch_size, dtype=tf.int32, seed=random.seed(datetime.now()))
            it = iter(self.unshuffled_test)
            for i in range(0, randomator):
                result = next(it)

            print('random number: ',randomator)
            # And cache it for next time
            # self._example = result

        return result

    def plot_renormalized(self, train_mean, train_std, model=None, plot_col='Temperature', max_subplots=3):

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
                predictions = model.predict(inputs)

            plt.plot(self.label_indices, ((
                predictions[n, :, label_col_index]*train_std[0])+train_mean[0]), '-', c='#ff7f0e', ms=1)
            plt.scatter(self.label_indices, ((predictions[n, :, label_col_index]*train_std[0])+train_mean[0]), marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

            plt.xticks(ticks=list(np.arange(288, 288+12)), labels=str_list)
            plt.title(model.model_name)

        plt.xlabel('minutes into the future')
        plt.show()

class Model():

    def __init__(self, window, resclale_factor):

        self.multi_val_performance = {}
        self.multi_performance = {}
        self.window = window
        self.rescale = resclale_factor

    def multi_lstm(self, OUT_STEPS, num_features_predicted, model_name, LoadModel=False):

        model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units].
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.LSTM(32, return_sequences=False, activation='tanh'),
            tf.keras.layers.Dropout(0.2),
            # Shape => [batch, out_steps*features].
            tf.keras.layers.Dense(OUT_STEPS*num_features_predicted,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features].
            tf.keras.layers.Reshape([OUT_STEPS, num_features_predicted])
        ])

        self.model_name = model_name
        self.model = model
        
        if LoadModel:

            self.load_pretrained_model()

    def double(self, OUT_STEPS, num_features_predicted, model_name, LoadModel=False):

        model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units].
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.LSTM(
                64, return_sequences=True, activation='tanh'),
            tf.keras.layers.LSTM(
                32, return_sequences=False, activation='tanh'),
            tf.keras.layers.Dropout(0.2),
            # Shape => [batch, out_steps*features].
            tf.keras.layers.Dense(OUT_STEPS*num_features_predicted,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features].
            tf.keras.layers.Reshape([OUT_STEPS, num_features_predicted])
        ])

        self.model_name = model_name
        self.model = model

        if LoadModel:

            self.load_pretrained_model()

    def bidirectional_lstm(self, OUT_STEPS, num_features_predicted, model_name, LoadModel=False):

        model = tf.keras.Sequential([
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(75, return_sequences=False)),
            tf.keras.layers.Dropout(0.01),
            tf.keras.layers.Dense(OUT_STEPS*num_features_predicted,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features].
            tf.keras.layers.Reshape([OUT_STEPS, num_features_predicted])

        ])

        self.model = model
        self.model_name = model_name

        if LoadModel:

            self.load_pretrained_model()

    def advanced_model(self, OUT_STEPS, num_features_predicted, model_name, LoadModel=False):

        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(
                64, kernel_size=6, activation='relu'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu'),
            tf.keras.layers.MaxPooling1D(2),
            # tf.keras.layers.LSTM(72, activation='relu', return_sequences=True),
            tf.keras.layers.LSTM(48, activation='relu',
                                 return_sequences=False),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(OUT_STEPS*num_features_predicted),
            tf.keras.layers.Reshape([OUT_STEPS,num_features_predicted])

        ])

        self.model = model
        self.model_name = model_name

        if LoadModel:

            self.load_pretrained_model()

    def multi_linear(self, OUT_STEPS, num_features_predicted, model_name, LoadModel=False):
        
        model = tf.keras.Sequential([
        # Take the last time-step.
        # Shape [batch, time, features] => [batch, 1, features]
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        # Shape => [batch, 1, out_steps*features]
            tf.keras.layers.Dense(OUT_STEPS*num_features_predicted,
                              kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([OUT_STEPS, num_features_predicted])
        ])

        self.model = model
        self.model_name = model_name

        if LoadModel:

            self.load_pretrained_model()

    def dense(self, OUT_STEPS, num_features_predicted, model_name, LoadModel=False):

        model = tf.keras.Sequential([
            # Take the last time step.
            # Shape [batch, time, features] => [batch, 1, features]
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            # Shape => [batch, 1, dense_units]
            tf.keras.layers.Dense(512, activation='relu'),
            # Shape => [batch, out_steps*features]
            tf.keras.layers.Dense(OUT_STEPS*num_features_predicted,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([OUT_STEPS, num_features_predicted])
        ])

        self.model = model
        self.model_name = model_name

        if LoadModel:

            self.load_pretrained_model()

    def cnn(self, OUT_STEPS, num_features_predicted, model_name, LoadModel=False):

        CONV_WIDTH = 3

        model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
            tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
            # Shape => [batch, 1, conv_units]
            tf.keras.layers.Conv1D(
                256, activation='relu', kernel_size=(CONV_WIDTH)),
            # Shape => [batch, 1,  out_steps*features]
            tf.keras.layers.Dense(OUT_STEPS*num_features_predicted,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([OUT_STEPS, num_features_predicted])
        ])

        self.model = model
        self.model_name = model_name
        
        if LoadModel:

            self.load_pretrained_model()

    def repeat_baseline(self, model_name):
        
        self.model = RepeatBaseline()
        self.model_name = model_name

    def compile_and_fit(self,EarlyStopping=True, TensorBoard=True, CheckPoint=False, epochs=20, patience=5):

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
        checkpoint_path = f"{self.model_name}/cp_.ckpt"

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./log')

        print('\n\n')
        print('Model name', self.model_name)

        if self.model_name == 'repeat_baseline':
            print('\nRepeat beaseline model, different compilation:\n')
            self.model.compile(loss=tf.keras.losses.MeanSquaredError(),
                               metrics=[tf.keras.metrics.MeanAbsoluteError()])

            history = None
        
        else:

            self.compile()

            callbacks=[]

            if (CheckPoint):
                callbacks.append(cp_callback)
                print('Saving weights')
            if (EarlyStopping):
                callbacks.append(early_stopping)
                print('Doing early stopping with patience =',patience)
            if (TensorBoard):
                callbacks.append(tensorboard_callback)
                print('Tensorboard callback available')

            print('\n')

            history = self.model.fit(self.window.train, epochs=epochs,
                        validation_data=self.window.val,
                        callbacks=callbacks)
        
        self.multi_val_performance[self.model_name] = self.model.evaluate(self.window.val)
        self.multi_performance[self.model_name] = self.model.evaluate(self.window.test, verbose=0)
        
        return history

    def compile(self):

        self.model.compile(loss=tf.keras.losses.MeanSquaredError(),
                           optimizer=tf.keras.optimizers.Adam(),
                           metrics=[tf.keras.metrics.MeanAbsoluteError()])

    def comparison_performances(self):

        x = np.arange(len(self.multi_performance))
        width = 0.3

        metric_name = 'mean_absolute_error'
        metric_index = self.model.metrics_names.index('mean_absolute_error')
        val_mae = [v[metric_index] for v in self.multi_val_performance.values()]
        test_mae = [v[metric_index] for v in self.multi_performance.values()]

        plt.figure(figsize=(9, 7))
        plt.bar(x - 0.17, val_mae, width, label='Validation')
        plt.bar(x + 0.17, test_mae, width, label='Test')
        plt.xticks(ticks=x, labels=self.multi_performance.keys(),
           rotation=45)
        plt.ylabel(f'MAE [°C] (average over all times and outputs)')
        _ = plt.legend()

        print(self.multi_val_performance)
        print(self.multi_performance)

        plt.show()

    def performance(self):

        self.multi_val_performance[self.model_name] = np.asarray(self.model.evaluate(self.window.val)) * self.rescale
        self.multi_performance[self.model_name] = np.asarray(self.model.evaluate(self.window.test, verbose=0)) * self.rescale

    def predict(self, inputs):
        return self.model(inputs)

    def load_pretrained_model(self):

        checkpoint_path = f"{self.model_name}/cp_.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        self.model.load_weights(latest)
        self.compile()

class RepeatBaseline(tf.keras.Model):
    def call(self, inputs):

        return inputs[:, -12:, 0]


