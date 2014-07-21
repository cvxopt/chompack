import unittest
import chompack as cp
from cvxopt import matrix,spmatrix,amd

class TestCspmatrix(unittest.TestCase):

    def setUp(self):
        I = list(range(17)) + [2,2,3,3,4,14,4,14,8,14,15,8,15,7,8,14,8,14,14,15,10,12,13,16,12,13,16,12,13,15,16,13,15,16,15,16,15,16,16]
        J = list(range(17)) + [0,1,1,2,2,2,3,3,4,4,4,5,5,6,6,6,7,7,8,8,9,9,9,9,10,10,10,11,11,11,11,12,12,12,13,13,14,14,15]
        self.A = spmatrix(matrix(range(len(I)),tc='d'),I,J,(17,17))
        self.symb = cp.symbolic(self.A, p = amd.order)

    def assertAlmostEqualLists(self,u,v):
        for ui,vi in zip(u,v): self.assertAlmostEqual(ui,vi)
        
    def test_cspmatrix(self):
        Ac = cp.cspmatrix(self.symb) + self.A
        self.assertTrue(Ac.is_factor is False)
        self.assertTrue(Ac.size[0] == Ac.size[1])
        self.assertTrue(Ac.size[0] == 17)

        diff = list((self.A - Ac.spmatrix(reordered=False,symmetric=False)).V)
        self.assertAlmostEqualLists(diff, len(diff)*[0.0])

        diff = list(cp.symmetrize(self.A) - Ac.spmatrix(reordered=False,symmetric=True))
        self.assertAlmostEqualLists(diff, len(diff)*[0.0])

        diff = list(cp.symmetrize(self.A)[self.symb.p,self.symb.p] - Ac.spmatrix(reordered=True,symmetric=True))
        self.assertAlmostEqualLists(diff, len(diff)*[0.0])

        diff = list(cp.tril(cp.symmetrize(self.A)[self.symb.p,self.symb.p]) - Ac.spmatrix(reordered=True,symmetric=False))
        self.assertAlmostEqualLists(diff, len(diff)*[0.0])
                
        diff = list((2*self.A -  (2*Ac).spmatrix(reordered=False,symmetric=False)).V)
        self.assertAlmostEqualLists(diff, len(diff)*[0.0])

        diff = list((2*self.A -  (Ac+Ac).spmatrix(reordered=False,symmetric=False)).V)
        self.assertAlmostEqualLists(diff, len(diff)*[0.0])

        Ac2 = Ac.copy()
        Ac2 += Ac
        diff = list((2*self.A -  Ac2.spmatrix(reordered=False,symmetric=False)).V)
        self.assertAlmostEqualLists(diff, len(diff)*[0.0])

        self.assertAlmostEqualLists(list(Ac.diag(reordered=False)), list(self.A[::18]))
        self.assertAlmostEqualLists(list(Ac.diag(reordered=True)), list(self.A[self.symb.p,self.symb.p][::18]))
                    
if __name__ == '__main__':
    unittest.main()
