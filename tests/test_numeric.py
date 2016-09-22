import unittest
import random
import chompack as cp
from cvxopt import matrix,spmatrix,amd,blas

class TestNumeric(unittest.TestCase):

    def setUp(self):
        n = 31
        nnz = int(round(0.15*n**2))
        random.seed(1)
        nz = matrix(random.sample(range(n**2), nnz), tc='i')
        self.A = cp.tril(spmatrix(matrix(range(1,nnz+1),tc='d')/nnz, nz%n, matrix([int(ii) for ii in nz/n]), (n,n)))\
          + spmatrix(10.0,range(n),range(n))
        self.symb = cp.symbolic(self.A, p = amd.order)

    def assertAlmostEqualLists(self,u,v):
        for ui,vi in zip(u,v): self.assertAlmostEqual(ui,vi)

    def test_dot(self):
        A = cp.cspmatrix(self.symb) + self.A
        B = cp.cspmatrix(self.symb) - 2*self.A
        self.assertAlmostEqual(2.0*blas.dot(A.blkval, B.blkval) - blas.dot(A.diag(),B.diag()), cp.dot(A,B))
                    
    def test_cholesky(self):
        L = cp.cspmatrix(self.symb) + self.A
        cp.cholesky(L)
        Lm = L.spmatrix(reordered=True)
        diff = list( (cp.tril(cp.perm(Lm*Lm.T,self.symb.ip)) - self.A).V )
        self.assertAlmostEqualLists(diff, len(diff)*[0.0])
        
    def test_llt(self):
        A = cp.cspmatrix(self.symb) + self.A
        cp.cholesky(A)
        cp.llt(A)
        Am = A.spmatrix(reordered=False)
        diff = list( (Am - self.A).V )
        self.assertAlmostEqualLists(diff, len(diff)*[0.0])

    def test_projected_inverse(self):
        Y = cp.cspmatrix(self.symb) + self.A
        Am = Y.spmatrix(symmetric=True,reordered=False)
        cp.cholesky(Y)
        cp.projected_inverse(Y)
        Ym = Y.spmatrix(symmetric=True,reordered=False)
        self.assertAlmostEqual((Ym.V.T*(Am.V))[0], self.symb.n)
        
    def test_completion(self):
        L = cp.cspmatrix(self.symb) + self.A
        cp.cholesky(L)
        L2 = L.copy()
        cp.projected_inverse(L2)
        cp.completion(L2, factored_updates = False)
        diff = list((L.spmatrix()-L2.spmatrix()).V)
        self.assertAlmostEqualLists(diff, len(diff)*[0.0])

    def test_completion_fu(self):
        L = cp.cspmatrix(self.symb) + self.A
        cp.cholesky(L)
        L2 = L.copy()
        cp.projected_inverse(L2)
        cp.completion(L2, factored_updates = True)
        diff = list((L.spmatrix()-L2.spmatrix()).V)
        self.assertAlmostEqualLists(diff, len(diff)*[0.0])
        
    def test_hessian(self):
        L = cp.cspmatrix(self.symb) + self.A
        cp.cholesky(L)
        Y = L.copy()
        cp.projected_inverse(Y)        
        U = cp.cspmatrix(self.symb) + 0.1*self.A
        cp.hessian(L, Y, U, adj = False, inv = False, factored_updates = False)
        cp.hessian(L, Y, U, adj = True, inv = False, factored_updates = False)
        cp.hessian(L, Y, U, adj = True, inv = True, factored_updates = False)
        cp.hessian(L, Y, U, adj = False, inv = True, factored_updates = False)
        diff = list((0.1*self.A-U.spmatrix(reordered=False,symmetric=False)).V)
        self.assertAlmostEqualLists(diff, len(diff)*[0.0])
        
    def test_hessian_fu(self):
        L = cp.cspmatrix(self.symb) + self.A
        cp.cholesky(L)
        Y = L.copy()
        cp.projected_inverse(Y)        
        U = cp.cspmatrix(self.symb) + 0.1*self.A
        cp.hessian(L, Y, U, adj = False, inv = False, factored_updates = True)
        cp.hessian(L, Y, U, adj = True, inv = False, factored_updates = True)
        cp.hessian(L, Y, U, adj = True, inv = True, factored_updates = True)
        cp.hessian(L, Y, U, adj = False, inv = True, factored_updates = True)
        diff = list((0.1*self.A-U.spmatrix(reordered=False,symmetric=False)).V)
        self.assertAlmostEqualLists(diff, len(diff)*[0.0])

    def test_trmm(self):
        L = cp.cspmatrix(self.symb) + self.A
        cp.cholesky(L)
        
        B = cp.eye(self.symb.n)
        cp.trmm(L, B)
        diff = list((L.spmatrix(reordered=True) - B[self.symb.p,self.symb.p])[:])
        self.assertAlmostEqualLists(diff, len(diff)*[0.0])

        B = cp.eye(self.symb.n)
        cp.trmm(L, B, trans = 'T')
        diff = list((L.spmatrix(reordered=True).T - B[self.symb.p,self.symb.p])[:])
        self.assertAlmostEqualLists(diff, len(diff)*[0.0])
    
    def test_trsm(self):
        L = cp.cspmatrix(self.symb) + self.A
        cp.cholesky(L)
        
        B = cp.eye(self.symb.n)
        Bt = matrix(B)[self.symb.p,:]
        Lt = matrix(L.spmatrix(reordered=True,symmetric=False))
        cp.trsm(L, B)
        blas.trsm(Lt,Bt)
        diff = list((B-Bt[self.symb.ip,:])[:])
        self.assertAlmostEqualLists(diff, len(diff)*[0.0])

        B = cp.eye(self.symb.n)
        Bt = matrix(B)[self.symb.p,:]
        Lt = matrix(L.spmatrix(reordered=True,symmetric=False))
        cp.trsm(L, B, trans = 'T')
        blas.trsm(Lt,Bt,transA='T')
        diff = list(B-Bt[self.symb.ip,:])[:]
        self.assertAlmostEqualLists(diff, len(diff)*[0.0])
        
    def test_pfcholesky(self):
        U = matrix(range(1,2*self.symb.n+1),(self.symb.n,2),tc='d')/self.symb.n
        alpha = matrix([1.2,-0.01])
        D = matrix(0.0,(2,2))
        D[::3] = alpha
        random.seed(1)
        V = matrix([random.random() for i in range(self.symb.n*3)],(self.symb.n,3))

        # PF Cholesky from spmatrix
        Lpf = cp.pfcholesky(self.A,U,alpha,p=amd.order)
        Vt = +V
        Lpf.trmm(Vt,trans='T')
        Lpf.trmm(Vt,trans='N')
        diff = list( (Vt - (cp.symmetrize(self.A) + U*D*U.T)*V)[:] )
        self.assertAlmostEqualLists(diff, len(diff)*[0.0])

        Lpf.trsm(Vt,trans='N')
        Lpf.trsm(Vt,trans='T')
        diff = list( (Vt-V)[:] )
        self.assertAlmostEqualLists(diff, len(diff)*[0.0])

        # PF Cholesky from cspmatrix factor
        L = cp.cspmatrix(self.symb) + self.A
        Lpf = cp.pfcholesky(L,U,alpha)
        Vt = +V
        Lpf.trmm(Vt,trans='T')
        Lpf.trmm(Vt,trans='N')
        diff = list( (Vt - (cp.symmetrize(self.A) + U*D*U.T)*V)[:] )
        self.assertAlmostEqualLists(diff, len(diff)*[0.0])

        Lpf.trsm(Vt,trans='N')
        Lpf.trsm(Vt,trans='T')
        diff = list( (Vt-V)[:] )
        self.assertAlmostEqualLists(diff, len(diff)*[0.0])
                
        
if __name__ == '__main__':
    unittest.main()
