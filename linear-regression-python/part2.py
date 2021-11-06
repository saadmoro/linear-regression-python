import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
#from datetime import datetime

training_data = pd.read_csv('PA1_train.csv')

#Part 0
#Remove ID feature from dataset
training_data = training_data.drop('id', 1)

#Split the date feature into month, day, year features.
training_data['date'] = pd.to_datetime(training_data['date'])
training_data['month'] = training_data['date'].dt.month
training_data['day'] = training_data['date'].dt.day
training_data['year'] = training_data['date'].dt.year 
del training_data['date']


#Actual part 1: Gradient Descent on normalized data
y = training_data['price']
del training_data['price']

#Remove zip code- not useful, was messing up later code by being too heavily weighted
del training_data['zipcode']

x = np.c_[training_data]

#Calculator for MSE
def mseCalc(w, X, y):
    N = len(y)
    pred = X.dot(w)
    mse = (1/N)*np.sum(np.square(pred - y))
    return mse

#Batch Gradient Descent
def gradientDescent(X, y, rate, iterations):
    N, d = np.shape(X)
    #w = np.random.uniform(0, 1, size=d)
    w = np.zeros(d)
    mse_history = np.zeros(iterations)

    for i in range(0, iterations):
        Xw = np.dot(X, w)
        Xt = X.transpose()
        sub = Xw - y
        gradient = np.dot(Xt, sub) / N
        w = w - rate * gradient
        mse_history[i] = mseCalc(w, X, y)

    return w, mse_history

numIterations = 10000
rate = 100

w, mse_history = gradientDescent(x, y, rate, numIterations)
print("Coefficient vector:")
print(w)

plt.scatter(range(numIterations), mse_history)
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.title("Iterations vs MSE for rate lambda = %f" % (rate))
plt.show()