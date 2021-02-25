
import numpy as np
from numpy import linalg as LA
from sklearn.preprocessing import normalize

class error:
    def __init__(self, problem, model_optimal, cost_optimal):
        self.pr = problem                                       ## problem class
        self.N = self.pr.N                                      ## total number of data samples
        self.X = self.pr.X_train                                ## feature vectors
        self.Y = self.pr.Y_train                                ## label vector
        self.theta_opt = model_optimal
        self.F_opt = cost_optimal

    def path_cls_error(self, iterates):
        iterates = np.array( iterates )
        Y_predict = np.matmul( self.X, iterates.T )
        error_matrix = np.multiply( Y_predict, self.Y[:,np.newaxis] ) < 0
        return np.sum( error_matrix, axis = 0 ) / self.N

    def point_cls_error(self, theta):
        Y_predict = np.matmul( self.X, theta )
        error = Y_predict * self.Y < 0
        return sum(error)/self.N
