from cvxopt import matrix, blas, lapack
from chompack.symbolic import cspmatrix
from chompack.misc import frontal_get_update

def projected_inverse(L):
    """
    Supernodal multifrontal projected inverse. The routine computes the projected inverse

    .. math::
         Y = P(L^{-T}L^{-1}) 

    where :math:`L` is a Cholesky factor. On exit, the argument :math:`L` contains the
    projected inverse :math:`Y`.

    :param L:                 :py:class:`cspmatrix` (factor)
    """

    assert isinstance(L, cspmatrix) and L.is_factor is True, "L must be a cspmatrix factor"

    n = L.symb.n
    snpost = L.symb.snpost
    snptr = L.symb.snptr
    chptr = L.symb.chptr
    chidx = L.symb.chidx

    relptr = L.symb.relptr
    relidx = L.symb.relidx
    blkptr = L.symb.blkptr
    blkval = L.blkval

    stack = []

    for k in reversed(list(snpost)):

        nn = snptr[k+1]-snptr[k]       # |Nk|
        na = relptr[k+1]-relptr[k]     # |Ak|
        nj = na + nn

        # invert factor of D_{Nk,Nk}
        lapack.trtri(blkval, offsetA = blkptr[k], ldA = nj, n = nn)

        # zero-out strict upper triangular part of {Nj,Nj} block (just in case!)
        for i in range(1,nn): blas.scal(0.0, blkval, offset = blkptr[k] + nj*i, n = i)   

        # compute inv(D_{Nk,Nk}) (store in 1,1 block of F)
        F = matrix(0.0, (nj,nj))
        blas.syrk(blkval, F, trans = 'T', offsetA = blkptr[k], ldA = nj, n = nn, k = nn)   

        # if supernode k is not a root node:
        if na > 0:

            # copy "update matrix" to 2,2 block of F
            Vk = stack.pop()
            lapack.lacpy(Vk, F, ldB = nj, offsetB = nn*nj+nn, m = na, n = na, uplo = 'L')

            # compute S_{Ak,Nk} = -Vk*L_{Ak,Nk}; store in 2,1 block of F
            blas.symm(Vk, blkval, F, m = na, n = nn, offsetB = blkptr[k]+nn,\
                      ldB = nj, offsetC = nn, ldC = nj, alpha = -1.0)

            # compute S_nn = inv(D_{Nk,Nk}) - S_{Ak,Nk}'*L_{Ak,Nk}; store in 1,1 block of F
            blas.gemm(F, blkval, F, transA = 'T', m = nn, n = nn, k = na,\
                      offsetA = nn, alpha = -1.0, beta = 1.0,\
                      offsetB = blkptr[k]+nn, ldB = nj)

        # extract update matrices if supernode k has any children
        for ii in range(chptr[k],chptr[k+1]):
            i = chidx[ii]
            stack.append(frontal_get_update(F, relidx, relptr, i))

        # copy S_{Jk,Nk} (i.e., 1,1 and 2,1 blocks of F) to blkval
        lapack.lacpy(F, blkval, m = nj, n = nn, offsetB = blkptr[k], ldB = nj, uplo = 'L')

    L._is_factor = False

    return
