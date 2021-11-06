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

#Normalize data

#Function to perform min-max normalization
def mmnormalize(vec):
    v = vec
    vec = (v - v.min())/(v.max() - v.min())
    return vec

normaldata = training_data

#These were what I considered to be numerical variables:
normaldata['bedrooms'] = mmnormalize(normaldata['bedrooms'])
normaldata['bathrooms'] = mmnormalize(normaldata['bathrooms'])
normaldata['sqft_living'] = mmnormalize(normaldata['sqft_living'])
normaldata['sqft_lot'] = mmnormalize(normaldata['sqft_lot'])
normaldata['floors'] = mmnormalize(normaldata['floors'])
normaldata['sqft_above'] = mmnormalize(normaldata['sqft_above'])
normaldata['sqft_basement'] = mmnormalize(normaldata['sqft_basement'])
normaldata['yr_built'] = mmnormalize(normaldata['yr_built'])
normaldata['yr_renovated'] = mmnormalize(normaldata['yr_renovated'])
normaldata['lat'] = mmnormalize(normaldata['lat'])
normaldata['long'] = mmnormalize(normaldata['long'])
normaldata['sqft_living15'] = mmnormalize(normaldata['sqft_living15'])
normaldata['sqft_lot15'] = mmnormalize(normaldata['sqft_lot15'])
normaldata['month'] = mmnormalize(normaldata['month'])
normaldata['day'] = mmnormalize(normaldata['day'])
normaldata['year'] = mmnormalize(normaldata['year'])

#Actual part 1: Gradient Descent on normalized data
y = normaldata['price']
del normaldata['price']

#Remove zip code- not useful, was messing up later code by being too heavily weighted
del normaldata['zipcode']

x = np.c_[normaldata]

#Calculator for MSE
def mseCalc(w, X, y):
    N = len(y)
    pred = X.dot(w)
    mse = (1/N)*np.sum(np.square(pred - y))
    return mse

#Batch Gradient Descent
#Takes X- data matrix; y - response variable, rate- training rate,
#iterations - maximum iterations allowable
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
        if np.linalg.norm(gradient) < 1:
            print("Iterations to convergence:")
            print(i)
            mse_history = mse_history[:i]
            break
    return w, mse_history

numIterations = 500000
rate = 0.0001

w, mse_history = gradientDescent(x, y, rate, numIterations)
print("Coefficient vector:")
print(w)


plt.scatter(range(len(mse_history)), mse_history)
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.title("Iterations vs MSE for rate lambda = %f" % (rate))
plt.show()

print("MSE for training data:")
print(mseCalc(w, x, y))

#TEST ON VALIDATION DATA- would have created this into a function to run for training
#and validation data, but oh well

validation_data = pd.read_csv('PA1_dev.csv')
validation_data = validation_data.drop('id', 1)

#Split the date feature into month, day, year features.
validation_data['date'] = pd.to_datetime(validation_data['date'])
validation_data['month'] = validation_data['date'].dt.month
validation_data['day'] = validation_data['date'].dt.day
validation_data['year'] = validation_data['date'].dt.year 
del validation_data['date']

validation_data['bedrooms'] = mmnormalize(validation_data['bedrooms'])
validation_data['bathrooms'] = mmnormalize(validation_data['bathrooms'])
validation_data['sqft_living'] = mmnormalize(validation_data['sqft_living'])
validation_data['sqft_lot'] = mmnormalize(validation_data['sqft_lot'])
validation_data['floors'] = mmnormalize(validation_data['floors'])
validation_data['sqft_above'] = mmnormalize(validation_data['sqft_above'])
validation_data['sqft_basement'] = mmnormalize(validation_data['sqft_basement'])
validation_data['yr_built'] = mmnormalize(validation_data['yr_built'])
validation_data['yr_renovated'] = mmnormalize(validation_data['yr_renovated'])
validation_data['lat'] = mmnormalize(validation_data['lat'])
validation_data['long'] = mmnormalize(validation_data['long'])
validation_data['sqft_living15'] = mmnormalize(validation_data['sqft_living15'])
validation_data['sqft_lot15'] = mmnormalize(validation_data['sqft_lot15'])
validation_data['month'] = mmnormalize(validation_data['month'])
validation_data['day'] = mmnormalize(validation_data['day'])
validation_data['year'] = mmnormalize(validation_data['year'])

yValid = validation_data['price']
del validation_data['price']

del validation_data['zipcode']

xValid = np.c_[validation_data]

print("MSE for validation data:")
print(mseCalc(w, xValid, yValid))