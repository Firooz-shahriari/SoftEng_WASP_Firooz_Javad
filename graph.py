"""
The class of random graphs,  Whether Directed or Undirected.
The graph should be strongle convex as its the assumption of paper
for convergence of the algorithms.

Input:  self.size: Number of  graph Nodes
        self.prob: probality of having edge between 2 nodes
Output: Adjacency matrix of the graph.
"""
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
import numpy as np
from numpy import linalg as LA
import math
import networkx as nx
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class Random:
    """ Initializetion of parameters """
    def __init__(self,number_of_nodes, prob):
        self.size = number_of_nodes
        self.prob = prob

    """ Directed Rnadom graph """
    def directed(self):
        edges = self.size*self.size
        indices = np.arange(edges)
        np.random.shuffle(indices)
        nonz = np.int(np.floor(edges*self.prob))
        non_zero_ind = indices[:nonz]
        Z = np.zeros(edges)
        Z[non_zero_ind] = 1.0
        D = Z.reshape(self.size,self.size)

        for i in range(self.size):
            if D[i][i] == 1.:
                D[i][i] = 0.

        """ Until here, we have the adjacency matrix of arandom directed graph,
            but we are not sure whether it is stringly connected or not! """
        graph = nx.from_numpy_matrix(D, create_using=nx.DiGraph)
        largest = max(nx.kosaraju_strongly_connected_components(graph), key=len)

        adj = np.zeros((len(largest), len(largest)))
        v = 0
        w = 0
        for i in largest:
            for j in largest:
                adj[v][w] = D[i][j]
                w +=1
            w = 0
            v +=1
        return adj

    """ Undirected Random Graph """
    def undirected(self):
        edges = self.size*self.size
        indices = np.arange(edges)
        np.random.shuffle(indices)
        nonz = np.int(np.floor(edges*self.prob))
        non_zero_ind = indices[:nonz]
        Z = np.zeros(edges)
        Z[non_zero_ind] = 1.0
        U = Z.reshape(self.size,self.size)
        for i in range(self.size):
            if U[i][i] == 1.:
                U[i][i] = 0.
            for j in range(self.size):
                if U[i][j] == 1:
                    U[j][i] = 1
        return U
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

"""
The class of exponential graphs: undirected and directed
As number of nodes increases exponentially, the degree at each node increases linearly.

Input:   self.size: number of graph nodes.
Output:  The adjacency matrix
"""

class Exponential:
    """ Initializatio of the number of nodes """
    def __init__(self,number_of_nodes):
        self.size = number_of_nodes

    """ Undirected Exponential Graph """
    def undirected(self):
        U = np.zeros( (self.size,self.size) )
        for i in range( self.size ):
            U[i][i] = 0
            neighbors = Expo_Neighbors(self.size)
            for j in neighbors:
                U[i][j] = 1
                U[j][i] = 1
        return U

    """ Directed Exponential Graph """
    def directed(self):
        D = np.zeros( (self.size,self.size) )
        for i in range( self.size ):
            D[i][i] = 0
            neighbors = Expo_Neighbors(self.size)
            for j in neighbors:
                D[i][j] = 1
        return D


def Expo_Neighbors(size):
    hops = np.array( range( int(math.log(size-1,2)) + 1 ) )
    return np.mod( i + 2 ** hops, size )
