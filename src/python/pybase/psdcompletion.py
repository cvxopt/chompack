from cvxopt import lapack, div, max, spmatrix, matrix
from chompack.symbolic import cspmatrix

def psdcompletion(A, reordered = True, **kwargs):
    """
    Maximum determinant positive semidefinite matrix completion. The
    routine takes a cspmatrix :math:`A` and returns the maximum determinant
    positive semidefinite matrix completion :math:`X` as a dense matrix, i.e.,

    .. math::
         P( X ) = A

    :param A:                 :py:class:`cspmatrix`
    :param reordered:         boolean
    """
    assert isinstance(A, cspmatrix) and A.is_factor is False, "A must be a cspmatrix"
    
    tol = kwargs.get('tol',1e-15)
    X = matrix(A.spmatrix(reordered = True, symmetric = True))

    symb = A.symb
    n = symb.n
    snptr = symb.snptr
    sncolptr = symb.sncolptr
    snrowidx = symb.snrowidx

    # visit supernodes in reverse (descending) order
    for k in range(symb.Nsn-1,-1,-1):

        nn = snptr[k+1]-snptr[k]
        beta = snrowidx[sncolptr[k]:sncolptr[k+1]]
        nj = len(beta)
        if nj-nn == 0: continue
        alpha = beta[nn:]
        nu = beta[:nn]
        eta = matrix([matrix(range(beta[kk]+1,beta[kk+1])) for kk in range(nj-1)] + [matrix(range(beta[-1]+1,n))])

        try:
            # Try Cholesky factorization first
            Xaa = X[alpha,alpha]
            lapack.potrf(Xaa)
            Xan = X[alpha,nu]
            lapack.trtrs(Xaa, Xan, trans = 'N')
            XeaT = X[eta,alpha].T
            lapack.trtrs(Xaa, XeaT, trans = 'N')

            # Compute update
            tmp = XeaT.T*Xan
            
        except:
            # If Cholesky fact. fails, switch to EVD: Xaa = Z*diag(w)*Z.T
            Xaa = X[alpha,alpha]
            w = matrix(0.0,(Xaa.size[0],1))
            Z = matrix(0.0,Xaa.size)
            lapack.syevr(Xaa, w, jobz='V', range='A', uplo='L', Z=Z)

            # Pseudo-inverse: Xp = pinv(Xaa)
            lambda_max = max(w)
            Xp = Z*spmatrix([1.0/wi if wi > lambda_max*tol else 0.0 for wi in w],range(len(w)),range(len(w)))*Z.T
                    
            # Compute update
            tmp = X[eta,alpha]*Xp*X[alpha,nu]

        X[eta,nu] = tmp
        X[nu,eta] = tmp.T

    if reordered:
        return X
    else:
        return X[symb.ip,symb.ip]

