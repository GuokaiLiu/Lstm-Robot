#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019-03-21

@author: liuguokai
"""
#%%
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from math import sin
from math import pi
from math import exp
from random import random
from random import randint
from random import uniform
from numpy import array
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import LSTM, Activation
from keras.layers import Dense, Dropout,TimeDistributed
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import scipy.io as sio
from keras.optimizers import Adam

# https://www.cnblogs.com/tiandsp/p/9687258.html
# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
# https://www.jianshu.com/p/38df71cad1f6
# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
'''
1-A 按照Xm(弹簧左端，液压缸的位置）
2-B Xl(弹簧右端，负载位置，Xm-Xl是弹簧压缩量，有正负）
3-C K（弹簧刚度）
4-D δK（弹簧刚度误差）
5-E m（负载质量）
6-F δm（负载质量误差）
7-G Fr(实际外界负载力）
8-H Fs(辨识的的负载力）

'''

if not os.path.exists('sheet1.npy'):
    dframe = pd.ExcelFile('Result.xlsx')
    print('Loading data will cost several minutes.')
    print('Data Loaded!')
    ##
    a = dframe.sheet_names
    b = dframe.parse(a[0],header=None)
    c = b.to_numpy()

    np.save('sheet1.npy',c)
else:
    print('sheet1.npy has been selected!')
d = np.load('sheet1.npy')
x = d[:,0:2] # Xm Xl
s = np.reshape(d[:,-2],(-1,1)) # Fr
y = np.reshape(d[:,-1],(-1,1))  # Fs

def Data_Vis():
    # Data Visualization
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax1.plot(x[:, 0])
    ax1.set_title('Xm')

    ax2 = fig.add_subplot(312)
    ax2.plot(x[:, 1])
    ax2.set_title('Xl')

    ax3 = fig.add_subplot(313)
    ax3.plot(y)
    ax3.set_title('Fs')
    plt.show()


# Data Preparation
look_back = 1
X_Raw = x
y_Raw = y
scalar_x = MinMaxScaler(feature_range=(0,1)).fit(X_Raw)
scalar_y = MinMaxScaler().fit(y_Raw)
X = scalar_x.transform(X_Raw)
y = scalar_y.transform(y_Raw)

X = np.reshape(X,(X.shape[0], 2, 1))

#%%

# generate damped sine wave in [0,1]
def generate_sequence(length, period, decay):
    return [0.5 + 0.5 * sin(2 * pi * i / period) * exp(-decay * i) for i in range(length)]


# generate input and output pairs of damped sine waves
def generate_examples(length, n_patterns, output):
    X, y = list(), list()
    for _ in range(n_patterns):
        p = randint(10, 20)
        d = uniform(0.01, 0.1)
        sequence = generate_sequence(length + output, p, d)
        X.append(sequence[:-output])
        y.append(sequence[-output:])
    X = array(X).reshape(n_patterns, length, 1)
    y = array(y).reshape(n_patterns, output)
    return X, y


def example():
    # X shape = (10000, 50,1)
    # y shape = (10000, 5)
    # configure problem
    length = 50
    output = 5
    # define model
    model = Sequential()
    # model.add(LSTM(20, return_sequences=True, input_shape=(length, 1)))
    model.add(LSTM(20, return_sequences=True, batch_input_shape=(None, 2, 1)))
    model.add(LSTM(20))

    model.add(Dense(output))
    model.compile(loss= 'mae' , optimizer= 'adam' )
    print(model.summary())
    # fit model
    X, y = generate_examples(length, 10000, output)
    history = model.fit(X, y, batch_size=10, epochs=1)
    # evaluate model
    X, y = generate_examples(length, 1000, output)
    loss = model.evaluate(X, y, verbose=0)
    print( 'MAE: %f' % loss)
    # prediction on new data
    X, y = generate_examples(length, 1, output)
    yhat = model.predict(X, verbose=0)
    pyplot.plot(y[0], label= 'y' )
    pyplot.plot(yhat[0], label= 'yhat' )
    pyplot.legend()
    pyplot.show()


def robot(x_train, y_train,x_test,y_test):
    learning_rate = 0.01
    opt = Adam(lr=learning_rate)
    batch_size = 1
    # define model
    # create and fit the LSTM network
    batch_size = 10
    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(batch_size, 2, 1), stateful=True, return_sequences=True))
    model.add(LSTM(4, batch_input_shape=(batch_size, 2, 1), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(100):
        model.fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
        model.reset_states()
    print(model.summary())
    X_Train = x_train
    y_Train = y_train
    # fit model
    # history = model.fit(X_Train, y_Train, batch_size=50, epochs=100,shuffle=False)
    # evaluate model
    X_Test = x_train
    y_Test = y_train
    # loss = model.evaluate(X_Test, y_Test, verbose=0)
    print( 'MAE: %f' % loss)
    # prediction on new data
    X_New = x_test
    y_New = y_test
    yhat = model.predict(X_New, verbose=0)

    return yhat,model
slice_s = 9000 # 2500[10-10.08]/2250[9.9-10]
slice_e = 9500
y_hat1,model = robot(X[slice_s:slice_e,:],y[slice_s:slice_e,:],X,y)
model.save('robot_model.h5')

y_hat2 = scalar_y.inverse_transform(y_hat1)


# fig = plt.figure(figsize=(8,20))
# ax1 = fig.add_subplot(211)
# ax1.plot(y)
# ax1.plot(y_hat)
# ax1.set_title('Start Point: 1, End Point: 10000.')
#
#
# ax2= fig.add_subplot(212)
# ax2.plot(y[3000:10000])
# ax2.plot(y_hat[3000:10000])
# ax2.set_title('Start Point: 3000, End Point: 10000.')


fig = plt.figure(figsize=(6,15))
ax1 = fig.add_subplot(411)
ax1.plot(y_Raw)
ax1.plot(y_hat2)
ax1.set_title('Predicted Data: Start Point: 1, End Point: 10000.')


ax2= fig.add_subplot(412)
# ax2.plot(y[3000:10000])
# ax2.plot(y_hat1[3000:10000])
ax2.plot(y[slice_s:slice_e])
ax2.plot(y_hat1[slice_s:slice_e])
ax2.set_title('Scaled Data: mu=0, std = 1, Start Point: 3000, End Point: 10000.')

ax3= fig.add_subplot(413)
ax3.plot(y_Raw[3000:10000])
ax3.plot(y_hat2[3000:10000])
ax3.set_title('Predicted Data: Start Point: 3000, End Point: 10000.')

ax4= fig.add_subplot(414)
ax4.plot(s[3000:10000])
ax4.plot(y_hat2[3000:10000])
ax4.set_title('Fr vs Fs_Predict')
plt.tight_layout()


# Save the result
adict = {}
# Save as .mat file for Matlab Execution
adict['Fr'] = s
adict['Fs'] = y
adict['y_hat'] = y_hat2
sio.savemat('robot.mat',adict)
