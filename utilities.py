import numpy as np
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
    adj = pseudo_adj - np.eye(size)
    row_sum = np.sum(adj, axis = 1)
    col_sum = np.sum(adj, axis = 0)
    l_in  = np.diag(row_sum) - adj
    l_out = np.diag(col_sum) - adj
    RS  = l_in  / (LDF*np.max(row_sum))     # Row stochastic
    CS  = l_out / (LDF*np.max(col_sum))     # Column Stochastic
    ZR  = np.eye(size) - RS                  # Zero Row-sum
    ZC  = np.eye(size) - CS                  # Zero Column-sum
    return ZR, ZC, RS, CS


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
