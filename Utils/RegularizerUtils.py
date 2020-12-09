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



'''Shaurya's portion follows, go through once'''

import numpy as np
import math
from AuxUtils import normalize, polynomial_features

#Lasso
class l1_regularization():
    def __init__(self, alpha):
        self.alpha = alpha
    def __call__(self, W):
        return self.alpha * np.linalg.norm(W)
    def grad(self, W):
        return self.alpha * np.sign(W)
#Ridge
class l2_regularization():
    def __init__(self, alpha):
        self.alpha = alpha
    def __call__(self, W):
        return self.alpha * 0.5 *  W.T.dot(W)
    def grad(self, W):
        return self.alpha * W

#Elastic Net
class l1_l2_regularization():
    def __init__(self, alpha, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
    def __call__(self, W):
        l1_comp = self.l1_ratio * np.linalg.norm(W)
        l2_comp = (1 - self.l1_ratio) * 0.5 * W.T.dot(W) 
        return self.alpha * (l1_comp + l2_comp)
    def grad(self, W):
        l1_comp = self.l1_ratio * np.sign(W)
        l2_comp = (1 - self.l1_ratio) * W
        return self.alpha * (l1_comp + l2_comp) 

class Regression(object):
    def __init__(self, n_iter, l_rate):
        self.n_iter = n_iter
        self.l_rate = l_rate
    def init_weights(self, n_features):
        limit = 1 / math.sqrt(n_features)
        self.W = np.random.uniform(-limit, limit, (n_features, ))
    def fit(self, x, y):
        x = np.insert(x, 0, 1, axis=1)
        self.training_errors = []
        self.init_weights(n_features=x.shape[1])
        # gradient descent
        for i in range(self.n_iter):
            y_hat = x.dot(self.W)
            # l2 loss
            mse = np.mean(0.5 * (y - y_hat)**2 + self.regularization(self.W))
            self.training_errors.append(mse)
            # l2 loss vs W
            grad_W = -(y - y_hat).dot(x) + self.regularization.grad(self.W)
            # update weights
            self.W -= self.l_rate * grad_W
    def predict(self, x):
        # biasing
        x = np.insert(x, 0, 1, axis=1)
        y_hat = x.dot(self.W)
        return y_hat

class linear_regression(Regression):
    def __init__(self, n_iter=100, l_rate=0.01, gradient_descent=True):
        self.gradient_descent = gradient_descent
        # No regularization
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(linear_regression, self).__init__(n_iter=n_iter, l_rate=l_rate)
    def fit(self, x, y):
        if not self.gradient_descent:
            x = np.insert(x, 0, 1, axis=1)
            U, S, V = np.linalg.svd(x.T.dot(x))
            S = np.diag(S)
            x_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
            self.W = x_sq_reg_inv.dot(x.T).dot(y)
        else:
            super(linear_regression, self).fit(x, y)

class lasso_regression(Regression):
    def __init__(self, degree, reg_factor, n_iter=3000, l_rate=0.01):
        self.degree = degree
        self.regularization = l1_regularization(alpha=reg_factor)
        super(lasso_regression, self).__init__(n_iter, l_rate)
    def fit(self, x, y):
        x = normalize(polynomial_features(x, degree=self.degree))
        super(lasso_regression, self).fit(x, y)
    def predict(self, x):
        x = normalize(polynomial_features(x, degree=self.degree))
        return super(lasso_regression, self).predict(x)

class polynomial_regression(Regression):
    def __init__(self, degree, n_iter=3000, l_rate=0.001):
        self.degree = degree
        # No regularization
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(polynomial_regression, self).__init__(n_iter=n_iter, l_rate=l_rate)
    def fit(self, x, y):
        x = polynomial_features(x, degree=self.degree)
        super(polynomial_regression, self).fit(x, y)
    def predict(self, x):
        x = polynomial_features(x, degree=self.degree)
        return super(polynomial_regression, self).predict(x)

class ridge_regression(Regression):
    def __init__(self, reg_factor, n_iter=1000, l_rate=0.001):
        self.regularization = l2_regularization(alpha=reg_factor)
        super(ridge_regression, self).__init__(n_iter, l_rate)

class polynomial_ridge_regression(Regression):
    def __init__(self, degree, reg_factor, n_iter=3000, l_rate=0.01, gradient_descent=True):
        self.degree = degree
        self.regularization = l2_regularization(alpha=reg_factor)
        super(polynomial_ridge_regression, self).__init__(n_iter, l_rate)
    def fit(self, x, y):
        x = normalize(polynomial_features(x, degree=self.degree))
        super(polynomial_ridge_regression, self).fit(x, y)
    def predict(self, x):
        x = normalize(polynomial_features(x, degree=self.degree))
        return super(polynomial_ridge_regression, self).predict(x)

class elastic_net(Regression):
    def __init__(self, degree=1, reg_factor=0.05, l1_ratio=0.5, n_iter=3000, l_rate=0.01):
        self.degree = degree
        self.regularization = l1_l2_regularization(alpha=reg_factor, l1_ratio=l1_ratio)
        super(elastic_net, self).__init__(n_iter, l_rate)
    def fit(self, x, y):
        x = normalize(polynomial_features(x, degree=self.degree))
        super(elastic_net, self).fit(x, y)
    def predict(self, x):
        x = normalize(polynomial_features(x, degree=self.degree))
        return super(elastic_net, self).predict(x)






