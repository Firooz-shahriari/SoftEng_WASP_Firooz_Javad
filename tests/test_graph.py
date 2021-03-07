import unittest
from graph import Random, Exponential
import numpy as np
from numpy import linalg as LA
import networkx as nx


class test_graph(unittest.TestCase):

    def setUp(self):
        self.A = Random(10, 0.5).undirected()
        self.B = Random(10, 0.8).undirected()
        self.C = Random(10, 0.5).directed()
        self.D = Random(10, 0.8).directed()

    def tearDown(self):
        pass

    """ As the graph is undirected its adjacency should be symmetric """
    def test_random_undirected_1(self):
        self.assertTrue( (np.array(self.A) == np.transpose(np.array(self.A))).all())
        self.assertTrue( (np.array(self.B) == np.transpose(np.array(self.B))).all())

    """ As the graph is Random, when the probality increase, we should have more edges """
    def test_random_undirected_2(self):
        self.assertGreaterEqual(np.count_nonzero(self.B), np.count_nonzero(self.A))


    """  As the graph is directed test is needed to check the graph is stringly onnected """
    def test_random_directed_1(self):
        CC = nx.from_numpy_matrix(self.C, create_using=nx.DiGraph)
        lenC = len(max(nx.kosaraju_strongly_connected_components(CC), key=len))
        DD = nx.from_numpy_matrix(self.D, create_using=nx.DiGraph)
        lenD = len(max(nx.kosaraju_strongly_connected_components(DD), key=len))
        self.assertEqual(lenC, (self.C).shape[0])
        self.assertEqual(lenD, (self.D).shape[0])

    """ As the graph is Random, when the probality increase, we should have more edges"""
    def test_random_directed_2(self):
        self.assertGreaterEqual(np.count_nonzero(self.D), np.count_nonzero(self.C))


    """ As the Expo graph is structured, test whther some samples must be  checked"""
    def test_Expo_directed_1(self):
        A = Exponential(3).directed()
        B = Exponential(4).directed()
        True_A = [[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]]
        True_B = [[0., 1., 1., 0.], [0., 0., 1., 1.], [1., 0., 0., 1.], [1., 1., 0., 0.]]
        self.assertTrue((A == True_A).all())
        self.assertTrue((B == True_B).all())

    """ If it is directed, it should be stronly connected"""
    def test_Expo_directed_2(self):
        C = Exponential(10).directed()
        CC = nx.from_numpy_matrix(C, create_using=nx.DiGraph)
        lenC = len(max(nx.kosaraju_strongly_connected_components(CC), key=len))
        self.assertEqual(lenC, C.shape[0])

    """ As the graph is undirected its adjacency should be symmetric """
    def test_Expo_undirected_1(self):
        E = Exponential(10).undirected()
        self.assertTrue( (E == np.transpose(E)).all())

if __name__ == '__main__':
    unittest.main()
