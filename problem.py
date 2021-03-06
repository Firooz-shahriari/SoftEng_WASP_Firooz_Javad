import numpy as np
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
"""
This class define the cost function or objective function that we want to minimize.
The function are log(cosh(.)), the constraint are also linear. 
The reason is the projection is easy to compute, which you can see in the following.
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

    """
    This method computes the function value. The function is written as separable 
    sum of several local objective functions
    """
    def F_val(self, theta):
        return np.sum(np.log10(np.cosh(np.matmul(self.C,theta)-self.e)))

    """
    This methods computes the gradient of function at each node
    """
    def localgrad(self, theta, idx):
        grad2 = (1 / np.cosh(np.inner(self.C[idx],theta[idx])-self.e[idx])) *\
                np.sinh(np.inner(self.C[idx], theta[idx])-self.e[idx]) * self.C[idx]
        return grad2


    """
    This method compute the gadient of the complete function:
    The function written as separable sum of nodes' functions 
    """
    def networkgrad(self, theta):
        grad = np.zeros((self.n, self.p))
        for i in range(self.n):
            grad[i] = self.localgrad(theta, i)
        return grad

    """
    This method computes the projection onto local constraints of each node.
    """
    def local_projection(self, idx, theta):
        tmp = np.inner(self.A[idx],theta[idx]) - self.b[idx]
        if tmp < 0:
            return theta[idx]
        else:
            return theta[idx] - tmp*( (self.A[idx]) / (np.linalg.norm(self.A[idx])) )

    """
    This method project into the complete constrint set
    """
    def network_projection(self, theta):
        proj = np.zeros((self.n, self.p))
        for i in range(self.n):
            proj[i] = self.local_projection(i, theta)
        return proj
