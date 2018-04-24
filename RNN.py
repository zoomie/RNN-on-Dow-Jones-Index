# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#This is going to be a very simple RNN
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

df = pd.read_csv('dow_jones_index.csv')

df['open'] = df['open'].map(lambda x: x.lstrip('$'))

training_set = df.iloc[:,3:4]

# Time to scale 
scaler = MinMaxScaler()
scaler.fit(training_set)
training_set = scaler.transform(training_set)


# Create the 60 time steps
X_train = []
y_train = []
for i in range(60 , len(df)):  
    X_train.append(training_set[i-60:i])
    y_train.append(training_set[i])

# Converting the lists into arrays
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 3, batch_size = 32)


# Predicting what happened!
X_test = X_train[689]
X_test = np.reshape(X_test, (1, 60, 1))
temp = X_test

result = []
for i in range(10):
    X_test = np.append(X_test, regressor.predict(temp))
    temp = X_test[-60:]
    temp = temp.reshape(1,60,1)
    
    
predicted_stock_price = scaler.inverse_transform(temp.reshape(60,1))












