import unittest
import chompack as cp
from cvxopt import matrix,spmatrix,amd,normal

class TestMrcompletion(unittest.TestCase):

    def setUp(self):
        I = list(range(17)) + [2,2,3,3,4,14,4,14,8,14,15,8,15,7,8,14,8,14,14,15,10,12,13,16,12,13,16,12,13,15,16,13,15,16,15,16,15,16,16]
        J = list(range(17)) + [0,1,1,2,2,2,3,3,4,4,4,5,5,6,6,6,7,7,8,8,9,9,9,9,10,10,10,11,11,11,11,12,12,12,13,13,14,14,15]
        self.A = spmatrix(1.0,I,J,(17,17))
        self.A[0,0] += 1.0
        self.A[1,1] += 1.0   
        self.symb = cp.symbolic(self.A, p = amd.order)

    def assertAlmostEqualLists(self,u,v):
        for ui,vi in zip(u,v): self.assertAlmostEqual(ui,vi)
        
    def test_mrcompletion(self):
        Ac = cp.cspmatrix(self.symb) + self.A
        Y = cp.mrcompletion(Ac)
        C = Y*Y.T
        Ap = cp.cspmatrix(Ac.symb)
        Ap.add_projection(C,beta=0.0,reordered=True)
        diff = list((Ac.spmatrix() - Ap.spmatrix()).V)
        self.assertAlmostEqualLists(diff, len(diff)*[0.0])

        U = normal(17,2)
        cp.syr2(Ac,U[:,0],U[:,0],alpha=0.5,beta=0.0)
        cp.syr2(Ac,U[:,1],U[:,1],alpha=0.5,beta=1.0)
        Y = cp.mrcompletion(Ac)
        Ap.add_projection(Y*Y.T,beta=0.0,reordered=True)
        diff = list((Ac.spmatrix() - Ap.spmatrix()).V)
        self.assertAlmostEqualLists(diff, len(diff)*[0.0])

        Ap.add_projection(U*U.T,beta=0.0,reordered=False)
        diff = list((Ac.spmatrix() - Ap.spmatrix()).V)
        self.assertAlmostEqualLists(diff, len(diff)*[0.0])
        
        
if __name__ == '__main__':
    unittest.main()
