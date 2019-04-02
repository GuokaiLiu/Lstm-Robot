from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM



# https://stackoverflow.com/questions/46102332/lstm-keras-api-predicting-multiple-outputs

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


data = np.load('sheet1.npy')
data = data[:,[0,1,-1]]
data[:,-1] = data[:,-1]-10

scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(data)


# plt.plot(scaled[:,-1])

n_steps_in = 3
n_steps_ou = 1
n_features = 2
n_obs = n_steps_in*n_features


reformed = series_to_supervised(scaled,n_steps_in,n_steps_ou) 	# with MinMaxScaler
print(reformed.shape) # m: L - (n_in + n_out) + 1, n = (n_in + n_out)*n_features


#%%
# reformed  = scaled
reformed = reformed.values

train = reformed[::2]  # Get all the odd lines
test = reformed[1::2]  # Get all the even lines

train_X = train[:, :n_obs]
test_X = test[:,:n_obs]

train_y = train[:,-1]
test_y = test[:,-1]


train_X = train_X.reshape((train_X.shape[0], n_steps_in, n_features))
test_X = test_X.reshape((test_X.shape[0], n_steps_in, n_features))


# train_y = train_y.reshape((train_y.shape[0], n_steps_ou, n_features))
# test_y = test_y.reshape((test_y.shape[0], n_steps_ou, n_features))
# train_y = train_y[:,0:2]
# test_y = test_y[:,0:2]
print('The training data shape is ', train_X.shape)
print('The testing data shape is ', test_X.shape)


# Evaluate Model
# design network

#%% Modeling

model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=200, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history

fig = plt.figure(1)

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()



#%% Evaluate Model
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_steps_in*n_features))

#%%
# invert scaling for forecast
inv_yhat = concatenate((test_X[:, 0:2], yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-1]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_X[:, 0:2], yhat), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,-1]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

fig = plt.figure(2)
plt.scatter(np.arange(len(inv_yhat)),inv_yhat,s=20,edgecolors='r',facecolor='none')
plt.scatter(np.arange(len(inv_y)),inv_y,s=0.5,edgecolors='b',)
plt.tight_layout()

#%%
fig = plt.figure(3)
plt.scatter(np.arange(len(inv_yhat)),inv_yhat-inv_y+10,s=20,edgecolors='r',facecolor='none')
plt.scatter(np.arange(len(inv_y)),np.ones((len(inv_y),1))*10,s=0.5,edgecolors='b')
plt.tight_layout()

