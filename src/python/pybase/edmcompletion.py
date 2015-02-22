from cvxopt import lapack, blas, div, max, spmatrix, matrix
from chompack.symbolic import cspmatrix

def edmcompletion(A, reordered = True, **kwargs):
    """
    Euclidean distance matrix completion. The routine takes an EDM-completable
    cspmatrix :math:`A` and returns a dense EDM :math:`X`
    that satisfies

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
        ne = len(eta)

        # Compute Yaa, Yan, Yea, Ynn, Yee
        Yaa = -0.5*X[alpha,alpha] - 0.5*X[alpha[0],alpha[0]]
        blas.syr2(X[alpha,alpha[0]], matrix(1.0,(nj-nn,1)), Yaa, alpha = 0.5)        

        Ynn = -0.5*X[nu,nu] - 0.5*X[alpha[0],alpha[0]]
        blas.syr2(X[nu,alpha[0]], matrix(1.0,(nn,1)), Ynn, alpha = 0.5)        

        Yee = -0.5*X[eta,eta] - 0.5*X[alpha[0],alpha[0]]
        blas.syr2(X[eta,alpha[0]], matrix(1.0,(ne,1)), Yee, alpha = 0.5)        
        
        Yan = -0.5*X[alpha,nu] - 0.5*X[alpha[0],alpha[0]]
        Yan += 0.5*matrix(1.0,(nj-nn,1))*X[alpha[0],nu]
        Yan += 0.5*X[alpha,alpha[0]]*matrix(1.0,(1,nn))

        Yea = -0.5*X[eta,alpha] - 0.5*X[alpha[0],alpha[0]]
        Yea += 0.5*matrix(1.0,(ne,1))*X[alpha[0],alpha]
        Yea += 0.5*X[eta,alpha[0]]*matrix(1.0,(1,nj-nn))
                
        # EVD: Yaa = Z*diag(w)*Z.T            
        w = matrix(0.0,(Yaa.size[0],1))
        Z = matrix(0.0,Yaa.size)
        lapack.syevr(Yaa, w, jobz='V', range='A', uplo='L', Z=Z)

        # Pseudo-inverse: Yp = pinv(Yaa)
        lambda_max = max(w)
        Yp = Z*spmatrix([1.0/wi if wi > lambda_max*tol else 0.0 for wi in w],range(len(w)),range(len(w)))*Z.T
                    
        # Compute update
        tmp = -2.0*Yea*Yp*Yan + matrix(1.0,(ne,1))*Ynn[::nn+1].T + Yee[::ne+1]*matrix(1.0,(1,nn))
        X[eta,nu] = tmp
        X[nu,eta] = tmp.T

    if reordered:
        return X
    else:
        return X[symb.ip,symb.ip]
