import problem 
import unittest
import numpy as np

class test_Problem(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        n  = 10
        p  = 5
        A1 = np.zeros((n,p))
        C1 = np.zeros((n,p))
        b  = np.zeros(n)
        e  = np.zeros(n)

        A2 = np.random.randn(n,p)
        C2 = np.random.randn(n,p)

        self.pr1 = problem(n, A1, b, C1, e)
        self.pr2 = problem(n, A2, b, C2, e)

        self.theta1 = np.random.randn(p)
        self.theta2 = np.zeros(p)

    def tearDown(self):
        pass

    """ linear functions are zero, the output should be zero """
    def test_F_val_1(self):
        self.assertEqual(self.pr1.F_val(self.theta1), 0)

    """ whenever theta and biases are zero, the output should be zero """
    def test_F_val_2(self):
        self.assertEqual(self.pr2.F_val(self.theta2), 0)

    """ random linear functions at random point """
    def test_F_val_3(self):
        self.assertEqual(self.pr2.F_val(self.theta1), 5.617305102469416)

    def test_localgrad(self):
        pass

    def test_networkGrad(self):
        pass

    def test_local_projection(self):
        pass

    def test_network_projection(self):
        pass


if __name__ == '__main__':
    unittest.main()