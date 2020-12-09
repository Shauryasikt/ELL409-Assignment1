import numpy as np 
from itertools import combinations

class PolynomialKernel()

    def __init__ (self, degree = 1, include_bias = 1):
        self.deg = degree 
        self.inc_bias = include_bias

    def kernelize (self, X):
        # X is the input features
        # degree is the required degree of the output
        # include_bias indicates whether a column of 1s should be present (indicates vertex)
        # we assume that X is an (n,m) array where n is the number of samples
        # and m is the number of features
        degree = self.deg
        include_bias = self.inc_bias
        np.asarray(X)
        n = X.shape[0]
        m = X.shape[1]
        final_features = include_bias + degree*m 
        deg = 2
        while (deg <= degrees):
                final_features = final_features + list(combinations(range(m), deg)).size()
        X_out = np.ones((n, final_features))
        pos = 0
        if (include_bias):
            X_out[:, 1:m+1] = X
            pos = m+1
        else:
            X_out [:, :m] = X
            pos = m
        deg = 2
        while (deg <= degree):
            combs_deg = list(combinations(range(m), deg))
            X_out[:, pos:pos+m] = X**deg
            pos = pos + m
            for comb in combs_deg:
                for num in comb:
                    X_out[:,pos] = X_out[:, pos]*X[:, num]
                pos = pos + 1
        return X_out



