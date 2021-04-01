import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from graph import Exponential, Random
from analysis import error
from problem import problem
import Algorithms as dopt
import matplotlib as mpl
import time
import os
mpl.rcParams['text.usetex'] = True
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# np.random.seed(0)
depoch      = 1000
n           = 10
dim         = 20
LDF         = 2           # Laplacian dividing factor
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
UG = Random(n, 0.55).directed()
adj = UG - np.eye(n)
row_sum = np.sum(adj, axis = 1)
col_sum = np.sum(adj, axis = 0)
l_in  = np.diag(row_sum) - adj
l_out = np.diag(col_sum) - adj
R222  = l_in  / (LDF*np.max(row_sum))
C222  = l_out / (LDF*np.max(col_sum))
R111  = np.eye(n) - R222
C111  = np.eye(n) - C222
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
theta_ddps = dopt.DDPS(lr_1, R111, C111, p, int( depoch ),theta_1, eps)
res_F_ddps = error_lr_1.cost_gap_path( np.sum(theta_ddps,axis = 1)/n)
fesgp_ddps = error_lr_1.feasibility_gap_syn2_1(np.sum(theta_ddps,axis = 1)/n)

theta_DAGP = dopt.ADDOPT(lr_1, R111 , C111  , step_size4, int( depoch ), theta_1)
res_F_DAGP = error_lr_1.cost_path( np.sum(theta_DAGP,axis = 1)/n)
fesgp_DAGP = error_lr_1.feasibility_gap_syn2_1(np.sum(theta_DAGP,axis = 1)/n)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# plt.rcParams['font.size'] = 28
plt.rcParams['axes.linewidth'] = 2
plt.rcParams["font.family"] = "Arial"
font = FontProperties()
font.set_size(16)
font2 = FontProperties()
font2.set_size(15)
mark_every = 5000
linewidth = 2

plt.figure(1, figsize=(7, 4))
plt.plot(res_F_DAGP,'-oy', markevery = mark_every, linewidth = linewidth)
plt.plot(res_F_ddps,'-^b', markevery = mark_every, linewidth = linewidth)
plt.grid(True)
plt.yscale('log')
# plt.tick_params(labelsize='large', width=3)
plt.tick_params(labelsize=16, width=3)
# plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 4),) # just copied
plt.ylabel(r'\textbf{Objective Value}', fontproperties=font2)
plt.xlabel(r'\textbf{Iterations}', fontproperties=font)
plt.legend((r'\textbf{ADDOPT}', r'\textbf{DDPS}'), prop={'size': 17}) #copied from .
filename = str(time.time()).split(".")[0]
timestamp = np.array([int(filename)])
filename = str(timestamp[0])
path = os.path.join(output_path, 'objective')
plt.savefig( path + ".pdf", format = 'pdf', dpi = 4000, pad_inches=0.05, bbox_inches ='tight')
plt.show(block = False)
#------------------------------------------------------------------------------

plt.figure(2, figsize=(7, 4))
plt.plot(fesgp_DAGP,'-oy', markevery = mark_every, linewidth = linewidth)
plt.plot(fesgp_ddps,'-^b', markevery = mark_every, linewidth = linewidth)
plt.grid(True)
plt.yscale('log')
# plt.tick_params(labelsize='large', width=3)
plt.tick_params(labelsize=16, width=3)
plt.ylabel(r'\textbf{Feasibility Error}', fontproperties=font2)
plt.xlabel(r'\textbf{Iterations}', fontproperties=font)
plt.legend((r'\textbf{ADDOPT}', r'\textbf{DDPS}'), loc = 1, prop={'size': 17}) #copied from .
filename = str(timestamp[0]+1)
path = os.path.join(output_path, 'feasibility')
plt.savefig( path + ".pdf", format = 'pdf', dpi = 4000, pad_inches=0.05, bbox_inches ='tight')
plt.show(block = False)
#------------------------------------------------------------------------------
plt.figure(5)
G = nx.from_numpy_matrix(np.matrix(UG), create_using=nx.DiGraph)
layout = nx.circular_layout(G)
nx.draw(G, layout)
path = os.path.join(output_path, 'graph')
plt.savefig( path + ".pdf", format = 'pdf', dpi = 4000, pad_inches=0, bbox_inches ='tight')
#------------------------------------------------------------------------------
