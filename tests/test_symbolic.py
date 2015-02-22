import unittest
import random
import chompack as cp
from cvxopt import matrix,spmatrix,amd

class TestSymbolic(unittest.TestCase):

    def setUp(self):
        I = list(range(17)) + [2,2,3,3,4,14,4,14,8,14,15,8,15,7,8,14,8,14,14,15,10,12,13,16,12,13,16,12,13,15,16,13,15,16,15,16,15,16,16]
        J = list(range(17)) + [0,1,1,2,2,2,3,3,4,4,4,5,5,6,6,6,7,7,8,8,9,9,9,9,10,10,10,11,11,11,11,12,12,12,13,13,14,14,15]
        self.A = spmatrix(matrix(range(len(I)),tc='d'),I,J,(17,17))

        n = 23
        nz = [1, 2, 10, 12, 13, 14, 15, 16, 19, 24, 48, 58, 71, 76, 81, 85, 91, 103, 108, 109, 111, 114, 116, 117, 118, 134, 143, 145, 161, 174, 178, 180, 183, 192, 194, 202, 203, 205, 212, 214, 219, 224, 226, 227, 228, 229, 235, 240, 241, 243, 247, 255, 256, 260, 269, 273, 275, 279, 280, 314, 315, 320, 321, 328, 340, 342, 344, 345, 349, 350, 357, 359, 370, 372, 375, 384, 385, 394, 399, 402, 411, 412, 419, 420, 422, 433, 439, 441, 447, 452, 454, 458, 460, 474, 476, 479, 481, 483, 485, 497, 506, 517, 519, 523, 526, 527]
        self.A_nc = cp.tril(spmatrix(matrix(range(1,len(nz)+1),tc='d')/len(nz),[ni%n for ni in nz],[int(ii/n) for ii in nz],(n,n)))\
          + spmatrix(10.0,range(n),range(n))

    def assertEqualLists(self,u,v):
        for ui,vi in zip(u,v): self.assertEqual(ui,vi)

    def assertAlmostEqualLists(self,u,v):
        for ui,vi in zip(u,v): self.assertAlmostEqual(ui,vi)
          
    def test_peo(self):
        self.assertTrue(cp.peo(self.A, matrix(range(17))))

    def test_symbolic(self):
        symb = cp.symbolic(self.A, p = None)

        self.assertEqual(symb.n, 17)
        self.assertEqual(symb.nnz, 56)
        self.assertEqual(symb.clique_number, 5)
        self.assertEqual(symb.Nsn, 9)
        self.assertEqual(symb.fill,(0,0))

        #p = amd.order(self.A)
        #symb = cp.symbolic(self.A, p = p)
        #self.assertEqual(list(symb.p), list(p))

        #symb = cp.symbolic(self.A, p = amd.order)
        #self.assertEqual(list(symb.p), list(p))
                    
    def test_merge(self):
        symb = cp.symbolic(self.A, p = None, merge_function = cp.merge_size_fill(0,0))
        self.assertEqual(symb.n, 17)
        self.assertEqual(symb.nnz, 56)
        self.assertEqual(symb.Nsn, 9)
        self.assertEqual(symb.fill,(0,0))
        self.assertEqual(symb.clique_number, 5)
        #self.assertEqual(symb.p, None)
        #self.assertEqual(symb.ip, None)

        symb = cp.symbolic(self.A, p = None, merge_function = cp.merge_size_fill(2,2))
        self.assertEqual(symb.n, 17)
        self.assertTrue(symb.nnz > 56)
        self.assertTrue(symb.Nsn < 9)
        self.assertTrue(symb.fill[0] >= 0)
        self.assertTrue(symb.fill[1] > 0)
        #self.assertEqual(symb.p, None)
        #self.assertEqual(symb.ip, None)

    def test_symbolic_nc(self):
        symb = cp.symbolic(self.A_nc, p = None)
        self.assertEqual(symb.n, 23)
        self.assertEqual(symb.nnz, 150)
        self.assertEqual(symb.clique_number, 12)
        self.assertEqual(symb.Nsn, 10)
        self.assertEqual(symb.fill,(73,0))
        #self.assertEqual(symb.p, None)
        #self.assertEqual(symb.ip, None)

        p = amd.order(self.A_nc)
        symb = cp.symbolic(self.A_nc, p = p)
        self.assertEqual(symb.n, 23)
        self.assertEqual(symb.nnz, 113)
        self.assertEqual(symb.clique_number, 9)
        self.assertEqual(symb.Nsn, 15)
        self.assertEqual(symb.fill,(36,0))
        #self.assertEqualLists(list(symb.p), list(p))
        #self.assertEqualLists(list(symb.p[symb.ip]),range(23))
        
    def test_merge_nc(self):
        symb = cp.symbolic(self.A_nc, p = None, merge_function = cp.merge_size_fill(0,0))
        self.assertEqual(symb.n, 23)
        self.assertEqual(symb.nnz, 150)
        self.assertEqual(symb.clique_number, 12)
        self.assertEqual(symb.Nsn, 10)
        self.assertEqual(symb.fill,(73,0))

        p = amd.order(self.A_nc)
        symb = cp.symbolic(self.A_nc, p = p, merge_function = cp.merge_size_fill(4,4))
        self.assertEqual(symb.n, 23)
        self.assertTrue(symb.nnz > 150)
        self.assertTrue(symb.Nsn < 10)
        self.assertTrue(symb.fill[0] >= 36)
        self.assertTrue(symb.fill[1] > 0)
        #self.assertEqualLists(list(symb.p), list(p))
        #self.assertEqualLists(list(symb.p[symb.ip]),range(23))


if __name__ == '__main__':
    unittest.main()
