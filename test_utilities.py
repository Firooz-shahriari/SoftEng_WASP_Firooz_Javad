import unittest
import utilities

class test_utilities(unittest.TestCase) :

    def setUp(self):
        self.name = 'Alg'

    def tearDown(self):
        pass

    def test_monitor(self):
        self.assertMultiLineEqual(self.name +' 10% completed',  utilities.monitor_testing(self.name, 9, 100))
        self.assertMultiLineEqual(self.name +' 90% completed',  utilities.monitor_testing(self.name, 89, 100))
        self.assertMultiLineEqual(self.name +' 100% completed', utilities.monitor_testing(self.name, 99, 100))

if __name__ == '__main__':
    unittest.main()
