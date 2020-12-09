import numpy as np 

def standardize (X):
	X = np.asarray(X)
	m = X.shape[1]
	#mean = np.array(m)
	#var = np.array(m)
	for i in range (m):
		mean = np.mean(X[:, i])
		var = np.mean((X[:, i] - mean)**2)
		X[:, i] = (X[:, i]-mean)/np.sqrt(var)
	return X






