from cvxopt import matrix, blas, lapack
from chompack.symbolic import cspmatrix
from chompack.misc import frontal_add_update

def llt(L):
    """
    Supernodal multifrontal Cholesky product:

    .. math::
         X = LL^T

    where :math:`L` is lower-triangular. On exit, the argument `L`
    contains the product :math:`X`.

    :param L:    :py:class:`cspmatrix` (factor)
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

    for k in snpost:

        nn = snptr[k+1]-snptr[k]       # |Nk|
        na = relptr[k+1]-relptr[k]     # |Ak|
        nj = na + nn

        # compute [I; L_{Ak,Nk}]*D_k*[I;L_{Ak,Nk}]'; store in F
        blas.trmm(blkval, blkval, side = "R", m = na, n = nn, ldA = nj,\
                  ldB = nj, offsetA = blkptr[k], offsetB = blkptr[k]+nn)
        F = matrix(0.0, (nj, nj))
        blas.syrk(blkval, F, n = nj, k = nn, offsetA = blkptr[k], ldA = nj)

        # if supernode k has any children, subtract update matrices 
        for _ in range(chptr[k],chptr[k+1]):
            Ui, i = stack.pop()
            frontal_add_update(F, Ui, relidx, relptr, i)

        # if supernode k is not a root node, push update matrix onto stack
        if na > 0:
            Uk = matrix(0.0,(na,na))
            lapack.lacpy(F, Uk, m = na, n = na, uplo = 'L', offsetA = nn*nj+nn, ldA = nj)
            stack.append((Uk,k))

        # copy leading Nk columns of F to blkval
        lapack.lacpy(F, blkval, m = nj, n = nn, ldA = nj, uplo = 'L',\
                     ldB = nj, offsetB = blkptr[k])

    L._is_factor = False

    return 
