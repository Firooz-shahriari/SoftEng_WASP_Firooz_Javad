import numpy as np

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
