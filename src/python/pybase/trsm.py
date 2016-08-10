from cvxopt import matrix, blas, lapack
from chompack.symbolic import cspmatrix

def trsm(L, B, alpha = 1.0, trans = 'N', nrhs = None, offsetB = 0, ldB = None):
    r"""
    Solves a triangular system of equations with multiple right-hand
    sides. Computes

    .. math::

       B &:= \alpha L^{-1} B  \text{ if trans is 'N'} \\
       B &:= \alpha L^{-T} B  \text{ if trans is 'T'} 

    where :math:`L` is a :py:class:`cspmatrix` factor.
    
    :param L:  :py:class:`cspmatrix` factor
    :param B:  matrix
    :param alpha:  float (default: 1.0)
    :param trans:  'N' or 'T' (default: 'N')   
    :param nrhs:   number of right-hand sides (default: number of columns in :math:`B`)
    :param offsetB: integer (default: 0)
    :param ldB:   leading dimension of :math:`B` (default: number of rows in :math:`B`)
    """
    
    assert isinstance(L, cspmatrix) and L.is_factor is True, "L must be a cspmatrix factor"
    assert isinstance(B, matrix), "B must be a matrix"

    if ldB is None: ldB = B.size[0]
    if nrhs is None: nrhs = B.size[1]
    assert trans in ['N', 'T']

    n = L.symb.n
    snpost = L.symb.snpost
    snptr = L.symb.snptr
    snode = L.symb.snode
    chptr = L.symb.chptr
    chidx = L.symb.chidx

    relptr = L.symb.relptr
    relidx = L.symb.relidx
    blkptr = L.symb.blkptr
    blkval = L.blkval

    p = L.symb.p
    if p is None: p = range(n)

    stack = []

    if trans is 'N':

        for k in snpost:

            nn = snptr[k+1]-snptr[k]       # |Nk|
            na = relptr[k+1]-relptr[k]     # |Ak|
            nj = na + nn

            # extract block from rhs
            Uk = matrix(0.0,(nj,nrhs))
            for j in range(nrhs):
                for i,ir in enumerate(snode[snptr[k]:snptr[k+1]]):
                    Uk[i,j] = alpha*B[offsetB + j*ldB + p[ir]]

            # add contributions from children
            for _ in range(chptr[k],chptr[k+1]):
                Ui, i = stack.pop()
                r = relidx[relptr[i]:relptr[i+1]]
                Uk[r,:] += Ui

            # if k is not a root node
            if na > 0:
                blas.gemm(blkval, Uk, Uk, alpha = -1.0, beta = 1.0, m = na, n = nrhs, k = nn,\
                          offsetA = blkptr[k]+nn, ldA = nj, offsetC = nn)
                stack.append((Uk[nn:,:],k))

            # scale and copy block to rhs
            blas.trsm(blkval, Uk, m = nn, n = nrhs, offsetA = blkptr[k], ldA = nj)
            for j in range(nrhs):
                for i,ir in enumerate(snode[snptr[k]:snptr[k+1]]):
                    B[offsetB + j*ldB + p[ir]] = Uk[i,j]

                
    else: # trans is 'T'

        for k in reversed(list(snpost)):
            
            nn = snptr[k+1]-snptr[k]       # |Nk|
            na = relptr[k+1]-relptr[k]     # |Ak|
            nj = na + nn

            # extract block from rhs and scale
            Uk = matrix(0.0,(nj,nrhs))
            for j in range(nrhs):
                for i,ir in enumerate(snode[snptr[k]:snptr[k+1]]):
                    Uk[i,j] = alpha*B[offsetB + j*ldB + p[ir]]
            blas.trsm(blkval, Uk, transA = 'T', m = nn, n = nrhs, offsetA = blkptr[k], ldA = nj)

            # if k is not a root node
            if na > 0:
                Uk[nn:,:] = stack.pop()
                blas.gemm(blkval, Uk, Uk, alpha = -1.0, beta = 1.0, m = nn, n = nrhs, k = na,\
                          transA = 'T', offsetA = blkptr[k]+nn, ldA = nj, offsetB = nn)
            
            # stack contributions for children
            for ii in range(chptr[k],chptr[k+1]):
                i = chidx[ii]
                stack.append(Uk[relidx[relptr[i]:relptr[i+1]],:])

            # copy block to rhs
            for j in range(nrhs):
                for i,ir in enumerate(snode[snptr[k]:snptr[k+1]]):
                    B[offsetB + j*ldB + p[ir]] = Uk[i,j]

    return
