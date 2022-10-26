# gemini-Project
One code to rule them all


import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from plot_keras_history import show_history, plot_history
import hyperopt 
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout, GRU, Bidirectional, Conv1D, MaxPooling1D
from hyperopt import hp, tpe, Trials, fmin, rand
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from hyperopt import fmin, tpe, hp, Trials, space_eval, STATUS_OK
import time
import threading
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import sklearn
from numba import jit
import seaborn as sns
import random
from sklearn.metrics import mean_squared_error
import math
import tensorflow_probability as tfp
import scipy
import multiprocessing
import plotly.figure_factory as ff
from tensorflow.keras.callbacks import EarlyStopping
tfd = tfp.distributions

excel=pd.read_excel(r"C:\Users\Francesco\OneDrive\Desktop\progetto icarus\data_bresso.xlsx")
df=pd.DataFrame(data=excel)
sns.set(style='darkgrid')
df['Temperature'] =  (df['Temperature'].str.split(' ', 1).str[0])
df['Dew Point'] =   (df['Dew Point'].str.split(' ', 1).str[0])
df['Speed'] =   (df['Speed'].str.split(' ', 1).str[0])



maxx=df.Temperature.max()
df['Temperature']=df['Temperature']/df['Temperature'].max()
df['Humidity']=df['Humidity']/df['Humidity'].max()
df['Dew Point']=df['Dew Point']/df['Dew Point'].max()
df['Speed']=df['Speed']/df['Speed'].max()
df['Gust']=df['Gust']/df['Gust'].max()
df['Pressure']=df['Pressure']/df['Pressure'].max()
df['Precip. Rate.']=df['Precip. Rate.']/df['Precip. Rate.'].max()
df['Precip. Accum.']=df['Precip. Accum.']/df['Precip. Accum.'].max()
df['Solar']=df['Solar']/df['Solar'].max()

df['Gust'] =   (df['Gust'].str.split(' ', 1).str[0])
df['Pressure'] =   (df['Pressure'].str.split(' ', 1).str[0])
df['Precip. Rate.'] =   (df['Precip. Rate.'].str.split(' ', 1).str[0])
df['Precip. Accum.'] =   (df['Precip. Accum.'].str.split(' ', 1).str[0])
df['Solar'] =   (df['Solar'].str.split(' ', 1).str[0])

df['Temperature'] =  (df['Temperature'].astype(float))
df['Dew Point'] =   (df['Dew Point'].astype(float))
df['Speed'] =   df['Speed'].astype(float)
df['Gust'] =   df['Gust'].astype(float)
df['Pressure'] =   df['Pressure'].astype(float)
df['Precip. Rate.'] =   df['Precip. Rate.'].astype(float)
df['Precip. Accum.'] =   df['Precip. Accum.'].astype(float)
df['Solar'] =   df['Solar'].astype(float)



df.drop(columns=['Wind', 'Time', 'Precip. Rate.', 'Precip. Accum.'], axis=1, inplace=True)
df.dropna(how='any', inplace=True)
x=12
y=48
y_, x_=[], []
interval=int((len(df)-(y+2*x))/x)
interval

for i in range(interval):
    y_.append(df.Temperature[x*i:y+x*i])

df.drop(columns=['Temperature'], axis=1, inplace=True)

for i in range(interval):
    x_.append(df[x*i:y+x*i])
    
lr=0.005
batch=16
epochs=500
opt = tf.keras.optimizers.Adam(learning_rate=lr)


y_=np.array(y_)
x_=np.array(x_)
x_.shape

xtrain=np.array(x_[0:9])
ytrain=np.array(y_[0:9])
xtest=np.array(x_[10:])
ytest=np.array(y_[10:])
xval=np.array(x_[9:10])
yval=np.array(y_[9:10])

def LSTM_model_tot(function):
    enco_deco=tf.keras.models.Sequential()
    # Encoder
    enco_deco.add(Bidirectional(LSTM(75, activation='tanh', return_sequences=True), input_shape=(48, 7)))
    enco_deco.add(Dropout(0.2))   
    enco_deco.add((Dense(units=1)))
    return enco_deco
start=time.time()

model=LSTM_model_tot('tanh')
model.compile(optimizer = opt, loss = 'mean_squared_error')
hist=model.fit(xtrain, ytrain, validation_data=(xval, yval), epochs=300, batch_size=7)
end=time.time()
print(end-start, 's')

ypredict=model.predict(xtest)

plot_history(hist)
plt.yscale('log')
plt.ylabel('log(Loss)')
plt.show()

ypredict=np.array(ypredict)
ypredict=ypredict.reshape(48)
ytest=ytest.reshape(48)

err=np.sqrt(sklearn.metrics.mean_squared_error(ytest[36:], ypredict[36:]))
print(err)

x=np.arange(0, 12)
sns.set(style="darkgrid")
fig, ax = plt.subplots(1, 2, figsize=(20, 8))

ax[0].plot(x, ypredict[36:]*maxx, color='b')
ax[0].plot(x, ytest[36:]*maxx, color='r')
ax[0].set_xlabel("index",fontsize=15)
ax[0].set_ylabel("outTemp",fontsize=15)
ax[0].fill_between(x, ypredict[36:]*maxx-err*maxx, ypredict[36:]*maxx+err*maxx, alpha=0.3)
ax[0].grid(True)
ax[0].legend(['test predictions', 'test values'])


ax[1].plot(x*5, abs(ytest[36:]-ypredict[36:])*maxx)
ax[1].set_ylabel('Error [°F]')
ax[1].set_xlabel('Time [min]')
ax[1].legend(['Last hour perdictions'])


plt.show()
