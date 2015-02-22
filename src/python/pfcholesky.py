import chompack as cp
from chompack.symbolic import cspmatrix, symbolic
from cvxopt import matrix, spmatrix, lapack

try:
    from chompack.cbase import pfchol, pftrmm, pftrsm
except:
    from math import sqrt
    def _ddrsv(n,trans,v,l,b,x):
        s = 0.0
        if trans=='N':
            for k in range(n):
                x[k] = (x[k] - v[k]*s)/l[k]
                s += b[k]*x[k]
        elif trans=='T':
            for k in range(n-1,-1,-1):
                x[k] = (x[k]-b[k]*s)/l[k]
                s += x[k]*v[k]
        else: raise ValueError("trans must be 'N' or 'T'")
        return

    def _ddrmv(n,trans,v,l,b,x):
        s = 0.0
        if trans=='N':
            for k in range(n):
                tmp = x[k]*b[k]
                x[k] = l[k]*x[k] + v[k]*s
                s += tmp
        elif trans=='T':
            for k in range(n-1,-1,-1):
                tmp = x[k]*v[k]
                x[k] = l[k]*x[k]+b[k]*s
                s += tmp
        else: raise ValueError("trans must be 'N' or 'T'")
        return

    def _dpftrf(n,k,a,V,L,B):
        for i in range(k):
            alpha = -a[i];
            v = V[:,i]
            l = L[:,i]
            b = B[:,i]
            
            for j in range(n):
                tmp = 1.0 - alpha*v[j]*v[j]
                if (tmp <= 0): return -1
                l[j] = sqrt(tmp)
                b[j] = -alpha*v[j]/l[j]
                alpha += b[j]*b[j]
            L[:,i] = l
            B[:,i] = b
                
            for j in range(i+1,k):
                vj = V[:,j]
                _ddrsv(n,'N',v,l,b,vj)
                V[:,j] = vj
        return
    
    def _dpfsv(n,k,trans,V,L,B,x):
        if trans=='N':
            for i in range(k):
                v = V[:,i]
                l = L[:,i]
                b = B[:,i]
                _ddrsv(n,trans,v,l,b,x);
                V[:,i] = v
        elif trans=='T':            
            for i in range(k-1,-1,-1):
                v = V[:,i]
                l = L[:,i]
                b = B[:,i]
                _ddrsv(n,trans,v,l,b,x)
                V[:,i] = v
        else: raise ValueError("trans must be 'N' or 'T'")
        return
        
    def _dpfmv(n,k,trans,V,L,B,x):
        if trans=='N':
            for i in range(k-1,-1,-1):
                v = V[:,i]
                l = L[:,i]
                b = B[:,i]
                _ddrmv(n,trans,v,l,b,x);
                V[:,i] = v
        elif trans=='T':            
            for i in range(k):
                v = V[:,i]
                l = L[:,i]
                b = B[:,i]
                _ddrmv(n,trans,v,l,b,x)
                V[:,i] = v
        else: raise ValueError("trans must be 'N' or 'T'")
        return
    
    def pfchol(alpha,V,L,B):
        _dpftrf(V.size[0],V.size[1],alpha,V,L,B)
        return

    def pftrsm(V,L,B,U,trans='N'):
        for i in range(U.size[1]):
            u = U[:,i]
            _dpfsv(U.size[0],L.size[1],trans,V,L,B,u)
            U[:,i] = u
        return

    def pftrmm(V,L,B,U,trans='N'):
        for i in range(U.size[1]):
            u = U[:,i]
            _dpfmv(U.size[0],L.size[1],trans,V,L,B,u)
            U[:,i] = u
        return
    
class pfcholesky(object):
    """
    Supernodal multifrontal product-form Cholesky factorization:

    .. math::
    
        X + V \mathrm{diag}(a) V^T = L_m \cdots L_1L_0 L_0^TL_1^T \cdots L_m^T

    where :math:`X = L_0L_0^T` is of order n and :math:`V` is n-by-m.

    :param X:       :py:class:`cspmatrix` or :py:class:`spmatrix`
    :param V:       n-by-m matrix
    :param a:       m-by-1 matrix (optional, default is vector of ones)
    :param p:       n-by-1 matrix (optional, default is natural ordering)
    """

    def __init__(self,X,V,a=None,p=None):

        self._n = X.size[0]
        self._V = +V        
        assert X.size[0] == X.size[1], 'X must be a square matrix'
        assert X.size[0] == V.size[0], 'V must have have the same number of rows as X'
        assert isinstance(V, matrix), 'V must be a dense matrix'
        
        if isinstance(X, cspmatrix) and X.is_factor is False:
            self._L0 = X.copy()            
            cp.cholesky(self._L0)
            cp.trsm(self._L0, self._V)
        elif isinstance(X, cspmatrix) and X.is_factor is True:
            self._L0 = X
            cp.trsm(self._L0, self._V)
        elif isinstance(X, spmatrix):
            symb = symbolic(X, p = p)
            self._L0 = cspmatrix(symb) + X
            cp.cholesky(self._L0)
            cp.trsm(self._L0, self._V)
        elif isinstance(X, matrix):
            raise NotImplementedError
        else:
            raise TypeError

        if a is None: a = matrix(1.0,(self._n,1))
        self._L = matrix(0.0,V.size)
        self._B = matrix(0.0,V.size)
        pfchol(a,self._V,self._L,self._B)
        return

    def __repr__(self):
        return "<%ix%i product-form Cholesky factor, r=%i, tc='%s'>"\
              % (self._L0.size[0],self._L0.size[0],self._V.size[1],self._L0.blkval.typecode) 

    def trsm(self,B,trans='N'):
        r"""
        Solves a triangular system of equations with multiple righthand
        sides. Computes

        .. math::

            B &:= L^{-1} B  \text{ if trans is 'N'}

           B &:= L^{-T} B  \text{ if trans is 'T'} 
        """
        
        if trans=='N':
            cp.trsm(self._L0,B)
            pftrsm(self._V,self._L,self._B,B,trans='N')
        elif trans=='T':
            pftrsm(self._V,self._L,self._B,B,trans='T')
            cp.trsm(self._L0,B,trans='T')
        elif type(trans) is str:
            raise ValueError("trans must be 'N' or 'T'")
        else:
            raise TypeError("trans must be 'N' or 'T'")
        return

    def trmm(self,B,trans='N'):
        r"""
        Multiplication with product-form Cholesky factor. Computes
    
        .. math::
    
            B &:= L B    \text{ if trans is 'N'}

            B &:= L^T B  \text{ if trans is 'T'}
        """
        
        if trans=='N':
            pftrmm(self._V,self._L,self._B,B,trans='N')
            cp.trmm(self._L0,B)            
        elif trans=='T':
            cp.trmm(self._L0,B,trans='T')
            pftrmm(self._V,self._L,self._B,B,trans='T')
        elif type(trans) is str:
            raise ValueError("trans must be 'N' or 'T'")
        else:
            raise TypeError("trans must be 'N' or 'T'")
        return

        
