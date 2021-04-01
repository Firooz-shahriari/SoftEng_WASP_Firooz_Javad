import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from graph import Exponential, Random
from analysis import error
from problem import problem
from check import graph_check
from utilities import graph_matrices, plot_figs, plot_graph
import Algorithms as dopt
import matplotlib as mpl
import time
import os
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# np.random.seed(0)
depoch      = 1000
n           = 10
dim         = 20
LDF         = 2           # Laplacian dividing factor
Edge_prob   = 0.55        # edge probability
output_path = "plots"

AA = np.random.randn(n, dim)
CC = np.random.randn(n, dim)
xx = np.random.randn(dim)
ee = np.random.randn(n)
bb = np.matmul(AA, xx) + 1*(np.random.rand(n))

# np.random.seed()
theta_1  = 1*np.random.randn(n,dim)
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# constrianed, my modification to my step_size
step_size4 = 0.05
rho4   = 0.05
alpha4 = 0.5
mu4    = 1.0
# DDPS p --> decaying power, eps: parameter
eps = 0.05
p   = 0.5
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
lr_1 = problem(n, AA, bb, CC, ee)
error_lr_1 = error(lr_1,np.zeros(n),0)
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
pseudo_adj     = Random(n, Edge_prob).directed()
ZR, ZC, RS, CS = graph_matrices(pseudo_adj, n, LDF)
graph_check(pseudo_adj, n)  # Introduce assertion
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
theta_ddps = dopt.DDPS(lr_1, RS, CS, p, int( depoch ),theta_1, eps)
res_F_ddps = error_lr_1.cost_gap_path( np.sum(theta_ddps,axis = 1)/n)
fesgp_ddps = error_lr_1.feasibility_gap(np.sum(theta_ddps,axis = 1)/n)

theta_DAGP = dopt.ADDOPT(lr_1, ZR, ZC, step_size4, int( depoch ), theta_1)
res_F_DAGP = error_lr_1.cost_path( np.sum(theta_DAGP,axis = 1)/n)
fesgp_DAGP = error_lr_1.feasibility_gap(np.sum(theta_DAGP,axis = 1)/n)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
plot_figs(res_F_DAGP, res_F_ddps, 16, 15, 5000, 2, 'Objective Value',   'Iterations', ('ADDOPT', 'DDPS'), os.path.join(output_path, 'objective'),   Block = False )
plot_figs(fesgp_DAGP, fesgp_ddps, 16, 15, 5000, 2, 'Feasibility Error', 'Iterations', ('ADDOPT', 'DDPS'), os.path.join(output_path, 'feasibility'), Block = False )
#------------------------------------------------------------------------------
plot_graph(pseudo_adj, os.path.join(output_path, 'graph'), Block = False)
#------------------------------------------------------------------------------
