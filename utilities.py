import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import time
import os
"""
This module is implemented to inform us about the progress of the algorithms,
during the experimets.
"""
def monitor(name,current,total):
    if (current+1) % (total/10) == 0:
        print ( name + ' %d%% completed' % int(100*(current+1)/total) )

def monitor_testing(name,current,total):
    if (current+1) % (total/10) == 0:
        return f'{name} {int(100*(current+1)/total)}% completed'

def graph_matrices(pseudo_adj, size, LDF):
    max_sum_in, l_in   = laplacian_matrix(pseudo_adj, 1, size)
    max_sum_out, l_out = laplacian_matrix(pseudo_adj, 0, size)
    RS  = l_in  / (LDF*max_sum_in)      # Row stochastic
    CS  = l_out / (LDF*max_sum_out)     # Column Stochastic
    ZR  = np.eye(size) - RS                  # Zero Row-sum
    ZC  = np.eye(size) - CS                  # Zero Column-sum
    return ZR, ZC, RS, CS

def laplacian_matrix(graph, AXis, size):
    adj  = graph - np.eye(size)
    sum  = np.sum(adj, axis = AXis)
    lap  = np.diag(sum) - adj
    return np.max(sum), lap

def plot_figs(y1, y2, fonts1, fonts2, mark_every, linewidth, Y_label, X_label, LEGEND, path, grid = True, Block = False):
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams["font.family"] = "Arial"
    font = FontProperties()
    font.set_size(fonts1)
    font2 = FontProperties()
    font2.set_size(fonts2)

    plt.figure(np.random.randint(1e9), figsize=(7, 4))
    plt.plot(y1,'-oy', markevery = mark_every, linewidth = linewidth)
    plt.plot(y2,'-^b', markevery = mark_every, linewidth = linewidth)
    plt.grid(grid)
    plt.yscale('log')
    plt.tick_params(labelsize=16, width=3)
    plt.ylabel(Y_label, fontproperties=font2)
    plt.xlabel(X_label, fontproperties=font)
    plt.legend(LEGEND, prop={'size': 17}) #copied from .
    filename = str(time.time()).split(".")[0]
    timestamp = np.array([int(filename)])
    filename = str(timestamp[0])
    plt.savefig( path + ".pdf", format = 'pdf', dpi = 4000, pad_inches=0.05, bbox_inches ='tight')
    plt.show(block = Block)

def plot_graph( graph, path, Block):
    plt.figure()
    G = nx.from_numpy_matrix(np.matrix(graph), create_using=nx.DiGraph)
    layout = nx.circular_layout(G)
    nx.draw(G, layout)
    plt.savefig( path + ".pdf", format = 'pdf', dpi = 4000, pad_inches=0, bbox_inches ='tight')
    plt.show(block = Block)
