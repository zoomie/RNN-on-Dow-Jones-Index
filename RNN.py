
# Simple RNN
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('dow_jones_index.csv')
# Removing the $ 
df['open'] = df['open'].map(lambda x: x.lstrip('$'))
# This model is only going to use opening price
training_set = df.iloc[:,3:4]

# Scaling the numbers
scaler = MinMaxScaler()
scaler.fit(training_set)
training_set = scaler.transform(training_set)


# Create the 60 time steps to that the model can be trained
# using the 60 previous prices to try determine the trend.
X_train = []
y_train = []
for i in range(60 , len(df)):  
    X_train.append(training_set[i-60:i])
    y_train.append(training_set[i])

# Converting the lists into arrays at this is the imput requirements for keras
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

# RNN with 3 hidden layers and drop out to stop overfitting
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Output layer
regressor.add(Dense(units = 1))

# Compiling the RNN 
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Training the model
regressor.fit(X_train, y_train, epochs = 3, batch_size = 32)


# Time to predict the future!
X_test = X_train[689]
X_test = np.reshape(X_test, (1, 60, 1))
temp = X_test

# This loop creates a prediction and then adds that to the set of data
# available so that it can predict the next day
for i in range(30):
    X_test = np.append(X_test, regressor.predict(temp))
    temp = X_test[-60:]
    temp = temp.reshape(1,60,1)
    
    
predicted_stock_price = scaler.inverse_transform(temp.reshape(60,1))
actual = scaler.inverse_transform(X_test.reshape(90,1)) 

# Visualising the results
plt.plot(list(range(60,90)), predicted_stock_price[30:60], color = 'red', label = 'Future Prediction')
plt.plot(actual[:60], color = 'blue', label = 'Actual price')
plt.title('RNN prediction of stock price')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()









