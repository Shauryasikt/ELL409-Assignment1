import numpy as np
import random
from math import log2
import matplotlib.pyplot as plt

# Calculate Mean Absolute Error
def mae_calc(actual, predicted):
	y = np.asarray(actual)
	yhat = np.asarray(predicted)
	error = y - yhat
	return np.mean(abs(error))

# Calculate Mean Squared Error
def mse_calc(actual, predicted):
	y = np.asarray(actual)
	yhat = np.asarray(predicted)
	error = y - yhat
	return np.mean(error**2)

# Calculate RMSE
def rmse_calc(actual, predicted):
	return np.sqrt(mse_calc(actual, predicted))

# Calculate R-squared
def r2_calc(actual, predicted):
	y = np.asarray(actual)
	yhat = np.asarray(predicted)
	error = y - yhat
	if sum((y-np.mean(y))**2)!=0:
		return 1-(sum(error**2)/sum((y-np.mean(y))**2))
	else:
		return 1

# Calculate cross entropy
def cross_entropy_calc(actual, predicted):

# Calculate log loss
def logloss_calc(actual, predicted):

# Calculate KL Divergence
def kl_div_calc(actual, predicted):
	actual = np.asarray(actual)
	predicted = np.asarray(predicted)
	return sum(actual[i] * log2(actual[i]/predicted[i]) for i in range(len(actual)))

# Calculate JS Divergence
def js_div_calc(actual, predicted):
	actual = np.asarray(actual)
	predicted = np.asarray(predicted)
	mid = 0.5*(actual + precited)
	return 0.5*kl_div_calc(actual, mid) + 0.5*kl_div_calc(predicted, mid)
