# Recurrent Neural Network



# Part 1 - Data Preprocessing

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the training set

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

#building dataset with 60 previous stock prices
X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

#reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Part 2 - Building the RNN
#importing libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#initializing the rnn
regressor = Sequential()
np.__version__
#adding the lstm layers and dropout regularization
regressor.add(LSTM(units = 50, return_sequ ences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#output layer
regressor.add(Dense(units = 1))

#compiling the rnn
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#fitting the model to the training data
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# Part 3 - Making the predictions and visualising the results

#importing the test data
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

#concatenating the training and test datasets
dataset_final = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

inputs = dataset_final[len(dataset_final) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)


X_test = []

for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#predicting
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


#plotting the prices

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Prices')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Prices')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
