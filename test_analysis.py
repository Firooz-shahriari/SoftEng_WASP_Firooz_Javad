from analysis import error
import numpy as np
import unittest
from problem import problem

class TestAnalysis(unittest.TestCase):

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
        self.n   = n
        self.p   = p
        self.theta1 = np.random.randn(p)
        self.theta2 = np.zeros(p)
        self.thet1  = np.random.randn(n,p)
        self.thet2  = np.zeros((n,p))

        self.error1 = error(self.pr1, np.zeros(self.p), 0)
        self.error2 = error(self.pr2, np.zeros(self.p), 0)

    def tearDown(self):
        pass

    def test_theta_gap_path(self):
        thet_norm = np.zeros(self.n)
        for i in range(self.n):
            thet_norm[i] = np.linalg.norm(self.thet1[i]) **2
        self.assertTrue((thet_norm == self.error1.theta_gap_path(self.thet1)).all())
        self.assertTrue((thet_norm == self.error2.theta_gap_path(self.thet1)).all())

    def test_cost_gap_point(self):
        self.assertTrue((self.pr1.F_val(self.theta1) == self.error1.cost_gap_point(self.theta1)).all())
        self.assertTrue((self.pr2.F_val(self.theta1) == self.error2.cost_gap_point(self.theta1)).all())

if __name__ == '__main__':
    unittest.main()
