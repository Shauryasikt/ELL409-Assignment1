import numpy as np 
from itertools import combinations

class PolynomialKernel():

    def __init__ (self, degree = 1, include_bias = 1):
        # degree is the required degree of the output
        # include_bias indicates whether a column of 1s should be present (indicates vertex)
        self.degree = degree 
        self.include_bias = include_bias

    def kernelize (self, X):
        d = self.degree
        ib = self.include_bias
        # X is the input features
        # we assume that X is an (n,m) array where n is the number of samples
        # and m is the number of features
        np.asarray(X)
        m = X.shape[1]
        final_features = ib + d*m 
        deg = 2
        while (deg <= d):
                final_features = final_features + list(combinations(range(m), deg)).size()
                deg += 1
        X_out = np.ones((n, final_features))
        pos = 0
        X_out[:, ib:m+ib] = X
        pos = m+ib
        deg = 2
        while (deg <= d):
            combs_deg = list(combinations(range(m), deg))
            X_out[:, pos:pos+m] = X**deg
            pos = pos + m
            for comb in combs_deg:
                for num in comb:
                    X_out[:,pos] = X_out[:, pos]*X[:, num]
                pos = pos + 1
            deg += 1
        return X_out
