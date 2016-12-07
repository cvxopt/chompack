from cvxopt import matrix, lapack, spmatrix
from chompack.symbolic import cspmatrix
from chompack.misc import frontal_get_update
from cvxopt import sqrt

def mrcompletion(A, reordered=True):
    """
    Minimum rank positive semidefinite completion. The routine takes a 
    positive semidefinite cspmatrix :math:`A` and returns a dense
    matrix :math:`Y` with :math:`r` columns that satisfies

    .. math::
         P( YY^T ) = A

    where 

    .. math::
         r = \max_{i} |\gamma_i|

    is the clique number (the size of the largest clique).
    
    :param A:                 :py:class:`cspmatrix`
    :param reordered:         boolean
    """

    assert isinstance(A, cspmatrix) and A.is_factor is False, "A must be a cspmatrix"

    symb = A.symb
    n = symb.n
    
    snpost = symb.snpost
    snptr = symb.snptr
    chptr = symb.chptr
    chidx = symb.chidx

    relptr = symb.relptr
    relidx = symb.relidx
    blkptr = symb.blkptr
    blkval = A.blkval

    stack = []
    r = 0 

    maxr = symb.clique_number
    Y = matrix(0.0,(n,maxr))       # storage for factorization
    Z = matrix(0.0,(maxr,maxr))    # storage for EVD of cliques
    w = matrix(0.0,(maxr,1))       # storage for EVD of cliques

    P = matrix(0.0,(maxr,maxr))    # storage for left singular vectors
    Q1t = matrix(0.0,(maxr,maxr))  # storage for right singular vectors (1)
    Q2t = matrix(0.0,(maxr,maxr))  # storage for right singular vectors (2)
    S = matrix(0.0,(maxr,1))       # storage for singular values

    V = matrix(0.0,(maxr,maxr))
    Ya = matrix(0.0,(maxr,maxr))
    
    # visit supernodes in reverse topological order
    for k in range(symb.Nsn-1,-1,-1):

        nn = snptr[k+1]-snptr[k]       # |Nk|
        na = relptr[k+1]-relptr[k]     # |Ak|
        nj = na + nn

        # allocate F and copy X_{Jk,Nk} to leading columns of F
        F = matrix(0.0, (nj,nj))
        lapack.lacpy(blkval, F, offsetA = blkptr[k], ldA = nj, m = nj, n = nn, uplo = 'L')

        # if supernode k is not a root node:
        if na > 0:
            # copy Vk to 2,2 block of F
            Vk = stack.pop()
            lapack.lacpy(Vk, F, offsetB = nn*nj+nn, m = na, n = na, uplo = 'L')

        # if supernode k has any children:
        for ii in range(chptr[k],chptr[k+1]):
            stack.append(frontal_get_update(F,relidx,relptr,chidx[ii]))

        # Compute factorization of F
        lapack.syevr(F, w, jobz='V', range='A', uplo='L', Z=Z, n=nj,ldZ=maxr)
        rk = sum([1 for wi in w[:nj] if wi > 1e-14*w[nj-1]])  # determine rank of clique k
        r = max(rk,r)                                         # update rank
        
        # Scale last rk cols of Z and copy parts to Yn
        for j in range(nj-rk,nj):
            Z[:nj,j] *= sqrt(w[j])
        In = symb.snrowidx[symb.sncolptr[k]:symb.sncolptr[k]+nn]
        Y[In,:rk] = Z[:nn,nj-rk:nj]

        # if supernode k is not a root node:
        if na > 0:
            # Extract data
            Ia = symb.snrowidx[symb.sncolptr[k]+nn:symb.sncolptr[k+1]]
            Ya[:na,:r] = Y[Ia,:r]
            V[:na,:rk] = Z[nn:nj,nj-rk:nj]
            V[:na,rk:r] *= 0.0            
            # Compute SVDs: V = P*S*Q1t and Ya = P*S*Q2t
            lapack.gesvd(V,S,jobu='A',jobvt='A',U=P,Vt=Q1t,ldU=maxr,ldVt=maxr,m=na,n=r,ldA=maxr)
            lapack.gesvd(Ya,S,jobu='N',jobvt='A',Vt=Q2t,ldVt=maxr,m=na,n=r,ldA=maxr)
            # Scale Q2t 
            for i in range(min(na,rk)):
                if S[i] > 1e-14*S[0]: Q2t[i,:r] = P[:na,i].T*Y[Ia,:r]/S[i]
            # Scale Yn            
            Y[In,:r] = Y[In,:r]*Q1t[:r,:r].T*Q2t[:r,:r]
                        
    if reordered:
        return Y[:,:r]
    else:
        return Y[symb.ip,:r]
