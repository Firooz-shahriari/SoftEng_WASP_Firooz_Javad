import networkx as nx
import numpy as np

def graph_check(graph_matrix, size):
    graph = nx.from_numpy_matrix(graph_matrix, create_using=nx.DiGraph)
    largest = max(nx.kosaraju_strongly_connected_components(graph), key=len)
    assert size == len(largest)
