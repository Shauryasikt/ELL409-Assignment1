import numpy as np
from Utils import ErrorMetricsUtils as err
from Utils import CorrectnessMetricUtils as cmu
from Utils import AuxUtils as auxu
import random
import time

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
        return self.alpha * 0.5 *  W.T@W
    def grad(self, W):
        return self.alpha * W

class LinearModel():  
    
    def __init__ (self, regression_degree = 50, l1_alpha = 0, l2_alpha = 0, loss = "mse", model_type = "linear", 
                  convergence = "Stochastic Gradiet Descent", include_bias = True, standardize = True, 
                  normalize = True, learning_type = "normalized", max_iter = 1e+5, learning_rate = 0.1, 
                  epsilon = 0.01, print_stuff = "prio"):
        self.l1_alpha = l1_alpha
        self.l2_alpha = l2_alpha
        self.loss = loss
        self.model_type = model_type
        self.convergence = convergence
        self.standardize = standardize
        self.normalize = normalize
        self.learning_type = learning_type
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.theta = None
        self.fitted = False
        self.print_stuff = print_stuff
        self.poly = auxu.PolynomialKernel(regression_degree, include_bias)
        if (self.model_type == "perceptron"):
            self.g = lambda x: np.piecewise(x, [x < 0, x >= 0], [0, 1])
        elif (self.model_type == "logistic"):
            self.g = lambda x: 1 / (1 + np.exp(-x))
        else:
            self.g = lambda x:x
    
    def call_convergence (self, X, y):
        if (self.convergence == "Stochatic Gradient Descent"):
            self.sch_grad_descent (X, y)
        elif (self.convergence == "Normal Equations"):
            self.normal_eqn (X, y)
        elif (self.convergence == "Newton's Method"):
             self.newton_method (X, y)
        else:
            self.bch_grad_descent (X, y)
        return
    
    def change_degree (self, degree):
        self.poly.degree = degree
    
    # General linear model
    def fit (self, X_train, y_train):
        #X_train = np.asarray(X)
        #y_train = np.asarray(y)
        if (X_train.ndim == 1):
            X_train = X_train[None, :]
        if (self.standardize):
            X_train = auxu.standardize(X_train)
        if (self.normalize):
            X_train = auxu.normalize(X_train)
        Xk_train = self.poly.kernelize(X_train)
        self.theta = np.zeros(Xk_train.shape[1])
        self.call_convergence(Xk_train, y_train)
        self.fitted = True
        return self.theta
    
    def predict (self, X_test):
        #X_test = np.asarray(X)
        if (X_test.ndim == 1):
            X_test = X_test[None, :]
        if (self.standardize):
            X_test = auxu.standardize(X_test)
        if (self.normalize):
            X_test = auxu.normalize(X_test)
        Xk_test = self.poly.kernelize(X_test)
        if (self.check_fitted(Xk_test)): 
            return self.g(Xk_test @ self.theta)
        else: 
            # some error statement
            return -1 
    
    def check_fitted (self, Xk_test):
        return self.fitted and self.theta.shape[0] == Xk_test.shape[1]
    
    def calc_loss (self, X, y):
        hypo = self.g (X @ self.theta)
        if (self.loss == "rmse"):
            return err.rmse_calc (y, hypo)
        elif (self.loss == "mae"):
            return err.mae_calc (y, hypo)
        elif (self.loss == "kld"):
            if (self.model_type == "logistic" or self.model_type == "perceptron"):
                return err.kl_divergence_calc (y, hypo)
            else:
                return float("Nan")
        elif (self.loss == "cross_entropy"):
            if (self.model_type == "logistic" or self.model_type == "perceptron"):
                return err.cross_entropy_calc (y, hypo)
            else:
                return float("Nan")
        else:
            # default case mse
            return err.mse_calc (y, hypo)
        
    def regularizer_neg_gradient(self):
        l1 = l1_regularization (self.l1_alpha)
        l2 = l2_regularization (self.l2_alpha)
        return - l1.grad(self.theta) - l2.grad(self.theta)
    
    def neg_gradient (self, X, y):
        hypo = self.g (X @ self.theta)
        neg_gradient = np.zeros(self.theta.shape)
        if (self.model_type == "logistic"):
            # only for binary classification
            if (self.loss == "mse"):
                neg_gradient = (X.T)@((y-hypo)*(hypo)*(1-hypo))
            elif (self.loss == "mae"):
                neg_gradient = (X.T)@(np.sign(y-hypo)*(hypo)*(1-hypo))
            else:
                # default case: log loss (same as cross entropy loss or kl divergence)
                neg_gradient = (X.T)@(y-hypo)
        elif (self.model_type == "perceptron"):
            # only for binary classification
            # 0-1 loss by default
            neg_gradient = (X.T)@(y-hypo)
        else:
            # Default model: linear
            if (self.loss == "mae"):
                neg_gradient = (X.T)@(np.sign(y-hypo))
            if (self.loss == "rmse"):
                rmse_err = self.calc_loss (X, y)
                neg_gradient = ((X.T)@(np.sign(y-hypo)))/rmse_err
            else: 
                # default case: mse loss
                neg_gradient = neg_gradient + (X.T)@(y-hypo)
        return neg_gradient + self.regularizer_neg_gradient()
        
    def descent_step (self, neg_gradient):
        if (self.learning_type == "normalized"):
            beta = self.learning_rate/np.linalg.norm(neg_gradient)
        else:
            beta = self.learning_rate/100
        self.theta = self.theta + beta*neg_gradient
        return beta*neg_gradient
        
    def print_state (self, iters, error, change_mean):
        if (self.print_stuff != "all"): return
        if (iters % 1000 == 0):
            print("After ", iters, " steps, the ", self.loss, " error is ", error, 
                  " and the change in theta was ", change_mean)
        return
        
    # General Batch Gradient Descent
    def bch_grad_descent (self, X, y):
        iters = 0
        while (iters < self.max_iter):
            neg_gradient = self.neg_gradient(X, y)
            change = self.descent_step (neg_gradient)
            err = self.calc_loss(X, y)
            self.print_state (iters, err, np.mean(change))
            if (np.linalg.norm(change) < self.epsilon): 
                break
            iters = iters + 1
        if (self.print_stuff != "none"):
            print ("Converged after ", iters, " steps")
        return
    
    def get_batch (self, X, y, batch_size):
        random.seed(time.time())
        index = random.randrange(X.shape[0] - batch_size)
        return X[index:index+batch_size, :], y[index:index+batch_size]
    
    # General Stochastic Gradient Descent
    def sch_grad_descent (self, X, y):
        batch_size = 49
        iters = 0
        while (iters < self.max_iter):
            X_sgd, y_sgd = self.get_batch (X, y, batch_size)
            neg_gradient = self.neg_gradient(X_sgd, y_sgd)
            change = self.descent_step (neg_gradient)
            err = self.calc_loss(X, y)
            self.print_state (iters, err, np.mean(change))
            if (np.linalg.norm(change) < self.epsilon): 
                break
            iters = iters + 1
        if (self.print_stuff != "none"):
            print ("Converged after ", iters, " steps")
        return
    
    # Newton's Method to solve regression 
    # Assumed MSE loss if linear, log loss if logistic and 0-1 loss if perceptron
    def newton_method (self, X, y):
        iters = 0;
        while (iters < self.max_iter):
            hypo = self.g (X @ self.theta)
            neg_gradient = X.T @ (y - hypo)
            hinv = np.linalg.pinv(X.T @ X)
            change = hinv @ neg_gradient
            if (np.linalg.norm (change, ord = 1) < self.epsilon): 
                break
            self.theta = self.theta + change
            iters = iters + 1
        return

    # Normal Equations to solve regression (assumed MSE loss function and linear/polynomial regression)
    def normal_eqn (self, X, y):
        self.theta = np.linalg.pinv(X.T @ X)@((X.T)@y)
        return
