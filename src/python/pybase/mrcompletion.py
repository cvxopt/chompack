from cvxopt import matrix, lapack, spmatrix
from chompack.symbolic import cspmatrix
from chompack.misc import frontal_get_update
from cvxopt import sqrt

def mrcompletion(A, reordered=True):
    """
    Minimum rank positive semidefinite completion. The routine takes a \
    positive semidefinite cspmatrix :math:`A` and returns a dense
    matrix :math:`Y` of size :math:`n \times r` that satisfies

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

    maxr = symb.clique_number
    Y = matrix(0.0,(n,maxr))     # storage for factorization
    Z = matrix(0.0,(maxr,maxr))  # storage for EVD of cliques
    w = matrix(0.0,(maxr,1))     # storage for EVD of cliques

    r = 0 
    
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
        lmax = max(w[:nj])
        rk = sum([1 for wi in w[:nj] if wi > 1e-14*lmax])  # determine rank of clique k
        r = max(rk,r)                                      # update max rank
        
        # scale last rk cols of Z and copy parts to Yn
        for j in range(nj-rk,nj):
            Z[:nj,j] *= sqrt(w[j])
        I = symb.snrowidx[symb.sncolptr[k]:symb.sncolptr[k]+nn]
        Y[I,:rk] = Z[:nn,nj-rk:nj]

                    
        # if supernode k is not a root node:
        if na > 0:
            V = matrix([[Z[nn:nj,nj-rk:nj]],[matrix(0.0,(na,r-rk))]])
            Ya = Y[symb.snrowidx[symb.sncolptr[k]+nn:symb.sncolptr[k+1]],:r]

            # Compute SVDs: V = P*S*Q1t and Ya = P*S*Q2t
            P = matrix(0.0,(na,na))
            Q1t = matrix(0.0,(r,r))
            Q2t = matrix(0.0,(r,r))
            S = matrix(0.0,(max(na,r),1))
            lapack.gesvd(V,S,jobu='A',jobvt='A',U=P,Vt=Q1t)
            lapack.gesvd(+Ya,S,jobu='N',jobvt='A',Vt=Q2t)
            for i in range(min(na,rk)):
                if S[i] > 1e-14*S[0]: Q2t[i,:] = P[:,i].T*Ya/S[i]

            # Scale Yn            
            Y[I,:r] = Y[I,:r]*Q1t.T*Q2t
                        
    if reordered:
        return Y[:,:r]
    else:
        return Y[symb.ip,:r]
