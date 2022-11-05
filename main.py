#!/usr/bin/env python
# -*- coding: utf-8 -*- #

from gemlib import *

"""
Put the path to `gemini-Project` of your local machine
"""
gemini_path = '/Users/matteocitterio/Documents/gemini-Project'

"""
This will automaticalluy retrieve the latest available `Bresso_updated_xxx` data.
"""
files = [i for i in os.listdir(gemini_path) if os.path.isfile(os.path.join(gemini_path, i)) and
         'Bresso_updated' in i]

"""
Load the fetched data into a `Pandas.DataFrame` object and track the maximum observed temperature.
This will be used to do the temperature conversion from [-\infty,1] data -> [min_temp, max_temp] [°F]
"""
path_name = (files[-1])
df, max_temp = load_data(path_name=path_name)
"""
Initialize a bool variable that flags plotting
"""
PlotCorrMatrix = False

if __name__ == "__main__":

    sns.set(style='darkgrid')

    #Reduce dataframe to last week data
    df = GetLastWeekData(df)

    # Correlation matrix
    if (PlotCorrMatrix):
        CorrMatrix(df)

    print(df)
    print(len(df))
    print(len(df)/7)

    #Da qui non capisco

    # loading data into 11 windows: x_ represent the future features whereas y_ will represent labels(variable that will be predicted)
    x=12
    y=144
    y_, x_=[], []
    interval=int((len(df)-y)/(x))
    print(interval)

    for i in range(interval):
        y_.append(df['Temperature'][x*i:y+x*i])

    df.drop(columns=['Temperature'], axis=1, inplace=True)

    for i in range(interval):
        x_.append(df[x*i:y+x*i])
    
    lr=0.0055
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    y_=np.array(y_)
    x_=np.array(x_)

    # Define the data split into train, valuation ans test set. The splitting is made in terms of windows number
    # Because of the input shape of the future lstm layer, we have to define test set with the same shape of the training set

    xtrain=x_[0:10]
    ytrain=y_[0:10]
    xtest=x_[2:]
    ytest=y_[11:]
    xval=x_[0:10][0:24]
    yval=y_[0:10][0:24]


    xtrain=xtrain.reshape(144, 10, 7)
    ytrain=ytrain.reshape(144, 10, 1)
    xval=xval.reshape(144, 10, 7)
    yval=yval.reshape(144, 10, 1)
    xtest=xtest.reshape(144, 10, 7)
    xtrain.shape
    # define the model as a single LSTM layer with 75 nodes and a dropout to avoid overfitting

    start=time.time()

    # compiling the previous model and fitting it in 76 epochs
    model=LSTM_model_tot('tanh')
    model.compile(optimizer = opt, loss = 'mean_squared_error')
    hist=model.fit(xtrain, ytrain, validation_data=(xval, yval), epochs=76, batch_size=3)
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
    ypredict=ypredict.reshape(1, 10, 144)
    ypredict=np.array(ypredict)
    ytest=ytest.reshape(144)

    #computing the error between predictions and test values using RMSE

    err=np.sqrt(sklearn.metrics.mean_squared_error(ytest[120:], ypredict[0][9][120:]))
    print(err)

    # Plotting the predictions and test trend 

    x=np.arange(0, 24)
    sns.set(style="darkgrid")
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    ax[0].plot(x*5, ypredict[0][9][120:]*maxx, color='b')
    ax[0].plot(x*5, ytest[120:]*maxx, color='r')
    ax[0].set_xlabel("Time",fontsize=15)
    ax[0].set_ylabel("Temperature",fontsize=15)
    ax[0].fill_between(x*5, ypredict[0][9][120:]*maxx-err*maxx, ypredict[0][9][120:]*maxx+err*maxx, alpha=0.3)
    ax[0].grid(True)
    ax[0].legend(['test predictions', 'test values'])



    ax[1].plot(x*5, abs(ytest[120:]-ypredict[0][9][120:])*maxx)
    ax[1].set_ylabel('Error [°F]')
    ax[1].set_xlabel('Time [min]')
    ax[1].legend(['Last hour perdictions'])


    plt.show()
    
    #HYPERPARAMETER OPTIMIZATION
    def test(model, xtest, ytest):
        ypredict=np.array(model.predict(xtest))
        ypredict=ypredict.reshape(1, 10, 144)
        ytest=ytest.reshape(144)
        loss= np.sqrt(sklearn.metrics.mean_squared_error(ytest[120:], ypredict[0][9][120:]))
        return loss

    def train(xtrain, ytrain, parameters):
        model=tf.keras.models.Sequential()
        # Encoder
        model.add(Bidirectional(LSTM(parameters['layer_size'], 
                                        activation='tanh', 
                                        return_sequences=True), 
                                    input_shape=(xtrain.shape[1], xtrain.shape[2])))
        model.add(Dropout(0.2))   
        model.add(Dense(1))
        
        adam=tf.keras.optimizers.Adam(learning_rate=parameters['learning_rate'])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(xtrain, 
                ytrain, 
                validation_data=(xval, yval),
                epochs=100, 
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
                'layer_size':hp.choice('layer_size', np.arange(50, 100, 5)), 
                'learning_rate': hp.uniform('learning_rate', 0.0001, 0.01), 
                'batch_size' : hp.choice('batch_size', np.arange(2, 8, 1))
                }
                
    trial=Trials()

    best=finding_best(search_space, trial)

    print(space_eval(search_space, best))
