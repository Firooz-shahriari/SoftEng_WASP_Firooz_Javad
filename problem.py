import numpy as np
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
"""
This class define the cost function or objective function that we want to minimize.
The function are log(cosh(.)), the constraint are also linear. 
The reason is the projection is easy to compute, which you can see in the following
"""
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

class problem():
    def __init__(self, n_agent, A, b, C, e):
        self.n = n_agent
        self.b = b
        self.A = A
        self.e = e
        self.C = C

        self.N = 1
        self.X_train = None
        self.Y_train = None
        self.p = A.shape[1]
        self.dim = A.shape[1]

    def F_val(self, theta):
        return np.sum(np.log10(np.cosh(np.matmul(self.C,theta)-self.e)))

    def localgrad(self, theta, idx):
        grad2 = (1 / np.cosh(np.inner(self.C[idx],theta[idx])-self.e[idx])) *\
                np.sinh(np.inner(self.C[idx], theta[idx])-self.e[idx]) * self.C[idx]
        return grad2

    def networkgrad(self, theta):
        grad = np.zeros((self.n, self.p))
        for i in range(self.n):
            grad[i] = self.localgrad(theta, i)
        return grad

    def grad(self, theta):
        pass
