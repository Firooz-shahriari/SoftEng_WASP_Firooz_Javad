"""
Defining Optimization algorithms that are used in the project as a module
--------------------------------------------------------------------------------
Two Algorithms:
    CGD: Centralized Gradient Decent
    ADD-Opt: exists in the literature
    DDPS: This algorithm can solve constrained problem
    I am not sure to consider this problem in this project or not!
--------------------------------------------------------------------------------
Inputs:
        pr: Object, that define the problem we are dealing with, problem.py.
        learning_rate: step size of the algorithms.
        K: Number of iterations
        theta_0: Initialization of the input.
        B1: Row-stochastic matrix
        B2: Column-stochastic matrix
        eps: intrinsic parameter of DDPS algorithm
        p: As DDPS use diminishing step_size, this value is used as d
        diminishing factor.
--------------------------------------------------------------------------------
Outputs:
        thata: list, the value of local variables at each iteration
        theta_opt: optimal point as CGD is used for comparison
        F_opt: optimal value as CGD is used for comparison
"""
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
import numpy as np
import copy as cp
import utilities as ut
from numpy import linalg as LA
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def Centralized_gradient_descent(pr,learning_rate,K,theta_0):
    theta = [theta_0]
    for k in range(K):
        theta.append( theta[-1] - learning_rate * pr.grad(theta[-1]) )
        ut.monitor('Centralized Gradient Descent',k,K)
    theta_opt = theta[-1]
    F_opt = pr.F_val(theta[-1])
    return theta, theta_opt, F_opt


def ADDOPT(prd,B1,B2,learning_rate,K,theta_0):
    theta = [ cp.deepcopy(theta_0) ]
    grad = prd.networkgrad( theta[-1] )
    tracker = cp.deepcopy(grad)
    Y = np.ones(B1.shape[1])
    for k in range(K):
        theta.append( np.matmul( B1, theta[-1] ) - learning_rate * tracker )
        grad_last = cp.deepcopy(grad)
        Y = np.matmul( B1, Y )
        YY = np.diag(Y)
        z = np.matmul( LA.inv(YY), theta[-1] )
        grad = prd.networkgrad( z )
        tracker = np.matmul( B2, tracker ) + grad - grad_last
        ut.monitor('ADDOPT', k ,K)
    return theta

def DDPS(prd, B1, B2, p, K, theta_0, eps):
    x = [ cp.deepcopy(theta_0) ]
    z = [ cp.deepcopy(theta_0) ]
    last_x = cp.deepcopy(theta_0)
    f_grad = prd.networkgrad( x[-1] )
    y = [ cp.deepcopy(f_grad) ]
    for k in range(K):
        alpha = (k+1)**(-p)
        z.append( np.matmul(B1, x[-1]) + eps*y[-1] - alpha*f_grad  )
        last_x = x [-1]
        x.append( prd.network_projection( z[-1]) )
        y.append( last_x - np.matmul(B1, last_x) + np.matmul(B2, y[-1]) - eps*y[-1] )
        f_grad = prd.networkgrad( x[-1] )
        ut.monitor('DDPS', k, K)
    return x
