import numpy as np
import pandas as pd

# Calculate the Gaussian probability distribution function for x
def Gaussian(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent

class GMM:
	# init function
	def __init__(self, n_clusters, max_iter, cluster_names=None):
        # number of clusters that the data will be split into
        self.n_clusters = n_clusters
        # number of iterations for the cluster										
        self.max_iter = max_iter
        # cluster names as either indexes or custom input
        if cluster_names == None:
            self.cluster_names = [f"cluster{index}" for index in range(self.n_clusters)]
        else:
            self.cluster_names = cluster_names
        # info as a list contains how much fraction of the dataset the cluster occupies
        self.info = [1/self.n_clusters for cluster in range(self.n_clusters)]

    # multivariate normal distribution formula
    def multivariate_normal(self, X, mean_rv, cov_matrix):
        # X => objective row vector
        # mean_rv => row vector with column-wise means
        # cov_matrix => 2D matrix with covariances for the features
        return (2*np.pi)**(-len(X)/2)*np.linalg.det(cov_matrix)**(-1/2)*np.exp(-np.dot(np.dot((X-mean_rv).T, np.linalg.inv(cov_matrix)), (X-mean_rv))/2)

    # model training
    def fit(self, X):
        # X => 2D vector: column = feature, row = data sample
        # Split the data into n clusters
        new_X = np.array_split(X, self.n_clusters)
        # calc of mean_rv and cov_matrix
        self.mean_rv = [np.mean(x, axis=0) for x in new_X]
        self.cov_matrices = [np.cov(x.T) for x in new_X]
        del new_X
        # EM
        for itern in range(self.max_iter):
            # E
            # r matrix init with probabilities in the rows
            self.r_matrix = np.zeros((len(X), self.n_clusters))
            # calc of r matrix
            for i in range(len(X)):
                for j in range(self.n_clusters):
                    self.r_matrix[i][j] = self.info[j]*self.multivariate_normal(X[i], self.mean_rv[j], self.cov_matrixes[j])
                    self.r_matrix[i][j] /= sum([self.info[k]*self.multivariate_normal(X[i], self.mean_rv[k], self.cov_matrixes[k]) for k in range(self.n_clusters)])
            N = np.sum(self.r_matrix, axis=0)
            # M
            # mean row vector init as a zero vector
            self.mean_rv = np.zeros((self.n_clusters, len(X[0])))
            # update mean row vector
            for i in range(self.n_clusters):
                for j in range(len(X)):
                    self.mean_rv[i] += self.r_matrix[j][i] * X[j]
            self.mean_rv = [1/N[k]*self.mean_rv[k] for k in range(self.n_clusters)]
            # cov_matrices list init
            self.cov_matrices = [np.zeros((len(X[0]), len(X[0]))) for k in range(self.n_clusters)]
            # update cov matrices
            for i in range(self.n_clusters):
                self.cov_matrices[i] = np.cov(X.T, aweights=(self.r[:, i]), ddof=0)
            self.cov_matrices = [1/N[j]*self.cov_matrices[j] for j in range(self.n_clusters)]
            # update info
            self.info = [N[i]/len(X) for i in range(self.n_clusters)]
        
    def predict(self, X):
        # # X => 2D vector: column = feature, row = data sample
        probs = []
        for n in range(len(X)):
            probs.append([self.multivariate_normal(X[n], self.mean_rv[i], self.cov_matrices[i]) for i in range(self.n_clusters)])
        cluster = []
        for prob in probs:
            cluster.append(self.cluster_names[prob.index(max(prob))])
        return cluster
