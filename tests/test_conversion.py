import unittest
import random
import chompack as cp
from cvxopt import matrix,spmatrix,amd

class TestConversion(unittest.TestCase):

    def setUp(self):
        pass
    
    def assertAlmostEqualLists(self,u,v):
        for ui,vi in zip(u,v): self.assertAlmostEqual(ui,vi)
            
    # Add test cases
    #
    #
    
if __name__ == '__main__':
    unittest.main()

