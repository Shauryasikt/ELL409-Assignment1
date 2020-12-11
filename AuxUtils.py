import math
import numpy as np 
import pandas as pd
from itertools import combinations_with_replacement

#think what this fn does
def shuffle_data(x, y, seed=None):
	if seed:
		np.random.seed(seed)
	index = np.arange(x.shape[0])
	np.random.shuffle(index)
	return x[index], y[index]

#think what this fn does
def separate_by_feature(x, feature, threshold):
	split_fn = None
	if isinstance(threshold, int) or isinstance(threshold, float):			#for int and float thresholds
		split_fn = lambda sample: sample[feature] >= threshold
	else:																	#for other "thresholds"
		split_fn = lambda sample: sample[feature] == thresholds
	x_0 = np.array([sample for sample in x if split_fn(sample)])
	x_1 = np.array([sample for sample in x if not split_fn(sample)])

	return np.array([x_0, x_1])

#think what this fn does
def train_test_split(x, y, test_size, shuffle=True, seed=None):
	if shuffle:
		x, y = shuffle_data(x, y, seed)
	splitter = len(y) - int(len(y) // (1/test_size))
	x_train, x_test = x[:splitter], x[splitter:]
	y_train, y_test = y[:splitter], y[splitter:]

	return x_train, x_test, y_train, y_test

#k-fold cross validation split
def cross_val_split(x, y, k, shuffle=True):
	if shuffle:
		x, y = shuffle_data(x, y)
	n_samples = len(y)
	rem = {}
	n_rem = (n_samples % k)
	if n_rem != 0:
		rem["x"] = x[-n_rem:]
		rem["y"] = y[-n_rem:]
		x = x[:-n_rem]
		y = y[:-n_rem]

	x_split = np.split(x, k)
	y_split = np.split(y, k)
	sets = []
	for i in range(k):
		x_test, y_test = x_split[i], y_split[i]
		x_train = np.concatenate(x_split[:i] + x_split[i + 1:], axis=0)
		y_train = np.concatenate(y_split[:i] + y_split[i + 1:], axis=0)
		sets.append([x_train, x_test, y_train, y_test])

	# remaining for training
	if n_rem != 0:
		np.append(sets[-1][0], rem["x"], axis=0)
		np.append(sets[-1][2], rem["y"], axis=0)

	return np.array(sets)

def normalize(x, axis=-1, order=2):
	aux = np.atleast_1d(np.linalg.norm(x, order, axis))
	aux[aux == 0] = 1
	x_nom = x / np.expand_dims(aux, axis) 
	return x_nom

def standardize(x):
	mean = x.mean(axis=0)
	std = x.std(axis=0)
	x_std = x
	for col in range(np.shape(x)[1]):
		if std[col]:
			x_std[:, col] = (x_std[:, col] - mean[col]) / std[col]
	return x_std

#vector to diagonal matrix
def to_diagonal(x):
	matrix = np.zeros((len(x), len(x)))
	for i in range(len(matrix[0])):
		matrix[i, i] = x[i]
	return matrix

#minkowski distance
def minkowski_distance(x1, x2, p):
	distance = 0
	k = float(1/p)
	for i in range(len(x1)):
		distance += pow((x1[i] - x2[i]), p)
	return pow(distance, k)

def polynomial_features(x, degree):
	n_samples, n_features = np.shape(x)
	combswr = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
	combinations = [item for sublist in combswr for item in sublist]
	n_output_features = len(combinations)
	x_new = np.empty((n_samples, n_output_features))
	
	for i, index_combs in enumerate(combinations):  
		x_new[:, i] = np.prod(x[:, index_combs], axis=1)

	return x_new

class PolynomialKernel():

	def __init__ (self, degree = 1, include_bias = True, interaction_only = False):
		# degree is the required degree of the output
		# include_bias indicates whether a column of 1s should be present (indicates vertex)
		self.degree = degree 
		self.include_bias = include_bias
		self.interaction_only = interaction_only

	def kernelize (self, X):
		X = np.asarray(X)
		if (X.ndim == 1):
			X = X[None, :]
		n_samples, n_features = np.shape(X)
		start = 0
		if (not self.include_bias): 
			start = 1
		combswr = [combinations_with_replacement(range(n_features), i) for i in range(start, self.degree + 1)]
		if (self.interaction_only):
			combswr = [combinations(range(n_features), i) for i in range(start, self.degree + 1)]
		combinations = [item for sublist in combswr for item in sublist]
		n_output_features = len(combinations)
		X_new = np.empty((n_samples, n_output_features))
		for i, index_combs in enumerate(combinations):  
			X_new[:, i] = np.prod(X[:, index_combs], axis=1)
		return X_new
