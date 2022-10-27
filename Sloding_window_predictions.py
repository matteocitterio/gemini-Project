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

# Loading dataset

excel=pd.read_excel(r"C:\Users\Francesco\OneDrive\Desktop\progetto icarus\data_bresso.xlsx")
df=pd.DataFrame(data=excel)
sns.set(style='darkgrid')

# Deleating units in dataset using str.plit function


df['Temperature'] =  (df['Temperature'].str.split(' ', 1).str[0])
df['Dew Point'] =   (df['Dew Point'].str.split(' ', 1).str[0])
df['Speed'] =   (df['Speed'].str.split(' ', 1).str[0])
df['Gust'] =   (df['Gust'].str.split(' ', 1).str[0])
df['Pressure'] =   (df['Pressure'].str.split(' ', 1).str[0])
df['Precip. Rate.'] =   (df['Precip. Rate.'].str.split(' ', 1).str[0])
df['Precip. Accum.'] =   (df['Precip. Accum.'].str.split(' ', 1).str[0])
df['Solar'] =   (df['Solar'].str.split(' ', 1).str[0])

# The previous function converted dataset into string, so We have to riconvert it in float

df['Temperature'] =  (df['Temperature'].astype(float))
df['Dew Point'] =   (df['Dew Point'].astype(float))
df['Speed'] =   df['Speed'].astype(float)
df['Gust'] =   df['Gust'].astype(float)
df['Pressure'] =   df['Pressure'].astype(float)
df['Precip. Rate.'] =   df['Precip. Rate.'].astype(float)
df['Precip. Accum.'] =   df['Precip. Accum.'].astype(float)
df['Solar'] =   df['Solar'].astype(float)

maxx=df.Temperature.max()

# Rescaling dataset from 0 to 1
df['Temperature']=df['Temperature']/df['Temperature'].max()
df['Humidity']=df['Humidity']/df['Humidity'].max()
df['Dew Point']=df['Dew Point']/df['Dew Point'].max()
df['Speed']=df['Speed']/df['Speed'].max()
df['Gust']=df['Gust']/df['Gust'].max()
df['Pressure']=df['Pressure']/df['Pressure'].max()
df['Precip. Rate.']=df['Precip. Rate.']/df['Precip. Rate.'].max()
df['Precip. Accum.']=df['Precip. Accum.']/df['Precip. Accum.'].max()
df['Solar']=df['Solar']/df['Solar'].max()

#Dropping data colums that are useless for the prediction of temperature 

df.drop(columns=['Wind', 'Time', 'Precip. Rate.', 'Precip. Accum.'], axis=1, inplace=True)
df.dropna(how='any', inplace=True)

# Define the window size, the slide and the number of window called interval
x=12
y=48
y_, x_=[], []
interval=int((len(df)-(y+2*x))/x)
interval

# loading data into 11 windows: x_ represent the future features whereas y_ will represent labels(variable that will be predicted)

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

# Define the data split into train, valuation ans test set. The splitting is made in terms of windows number

xtrain=np.array(x_[0:9])
ytrain=np.array(y_[0:9])
xtest=np.array(x_[10:])
ytest=np.array(y_[10:])
xval=np.array(x_[9:10])
yval=np.array(y_[9:10])

# define the model as a single LSTM layer with 75 nodes and a dropout to avoid overfitting

def LSTM_model_tot(function):
    enco_deco=tf.keras.models.Sequential()
    # Encoder
    enco_deco.add(Bidirectional(LSTM(75, activation='tanh', return_sequences=True), input_shape=(48, 7)))
    enco_deco.add(Dropout(0.2))   
    enco_deco.add((Dense(units=1)))
    return enco_deco
start=time.time()

# compiling the previous model and fitting it in 300 epochs(200 is fine too)

model=LSTM_model_tot('tanh')
model.compile(optimizer = opt, loss = 'mean_squared_error')
hist=model.fit(xtrain, ytrain, validation_data=(xval, yval), epochs=300, batch_size=7)
end=time.time()
print(end-start, 's')

# Plotting validation and training loss trend in terms of epochs

plot_history(hist)
plt.yscale('log')
plt.ylabel('log(Loss)')
plt.show()


# Generating predictions using test set

ypredict=model.predict(xtest)

# Reshaping predictions and test set from (1, 48) to (40)
ypredict=np.array(ypredict)
ypredict=ypredict.reshape(48)
ytest=ytest.reshape(48)

#computing the error between predictions and test values using RMSE

err=np.sqrt(sklearn.metrics.mean_squared_error(ytest[36:], ypredict[36:]))
print(err)

# Plotting the predictions and test trend 

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

#Plotting the difference trend in terms of time

ax[1].plot(x*5, abs(ytest[36:]-ypredict[36:])*maxx)
ax[1].set_ylabel('Error [°F]')
ax[1].set_xlabel('Time [min]')
ax[1].legend(['Last hour perdictions'])


plt.show()


 
#HYPERPARAMETER OPTIMIZATION

def test(model, xtest, ytest):
    ypredict = np.array(model.predict(xtest))
    ypredict=ypredict.reshape(48)
    ytest=ytest.reshape(48)
    loss= np.sqrt(sklearn.metrics.mean_squared_error(ytest, ypredict))
    return loss

def train(xtrain, ytrain, parameters):
    model=tf.keras.models.Sequential()
    # Encoder
    model.add(Bidirectional(LSTM(parameters[layer_size], activation='tanh', return_sequences=True), input_shape=(48, 7)))
    model.add(Dropout(0.2))   
    model.add((Dense(units=1)))
    adam=tf.keras.optimizers.Adam(learning_rate=parameters['learning_rate'])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(xtrain, 
              ytrain, 
              validation_data=(xval, yval), 
              epochs=300, 
              batch_size=parameters['batch_size'])
    return model



def finding_best(search_space, trial):    
    return fmin(fn=hyper_func, space=search_space, algo=tpe.suggest, max_evals=50, trials=trial)


#Function to miminize
def hyper_func(params):

    model = train(xtrain, ytrain, params)
    loss = test(model, xtest, ytest)
    return{'loss': loss, 'status': STATUS_OK}

#Hyperparameter space: we want to optimize leraning rate and nodes number

search_space={
              'layer_size':hp.choice('layer_size', np.arange(30, 80, 5)), 
              'learning_rate': hp.uniform('learning_rate', 0.0001, 0.01), 
              'batch_size': hp.choice('batch_size', np.arange(4, 11, 2))
              }
              
trial=Trials()

best=finding_best(search_space, trial)

print(space_eval(search_space, best))
