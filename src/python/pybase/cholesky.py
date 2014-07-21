from cvxopt import matrix, blas, lapack
from chompack.symbolic import cspmatrix
from chompack.misc import frontal_add_update

def cholesky(X):
    """
    Supernodal multifrontal Cholesky factorization:

    .. math::
         X = LL^T

    where :math:`L` is lower-triangular. On exit, the argument :math:`X`
    contains the Cholesky factor :math:`L`.

    :param X:    :py:class:`cspmatrix`
    """

    assert isinstance(X, cspmatrix) and X.is_factor is False, "X must be a cspmatrix"

    n = X.symb.n
    snpost = X.symb.snpost
    snptr = X.symb.snptr
    chptr = X.symb.chptr
    chidx = X.symb.chidx

    relptr = X.symb.relptr
    relidx = X.symb.relidx
    blkptr = X.symb.blkptr
    blkval = X.blkval

    stack = []

    for k in snpost:

        nn = snptr[k+1]-snptr[k]       # |Nk|
        na = relptr[k+1]-relptr[k]     # |Ak|
        nj = na + nn                   

        # build frontal matrix
        F = matrix(0.0, (nj, nj))
        lapack.lacpy(blkval, F, offsetA = blkptr[k], m = nj, n = nn, ldA = nj, uplo = 'L')

        # add update matrices from children to frontal matrix
        for i in range(chptr[k+1]-1,chptr[k]-1,-1):
            Ui = stack.pop()
            frontal_add_update(F, Ui, relidx, relptr, chidx[i])

        # factor L_{Nk,Nk}
        lapack.potrf(F, n = nn, ldA = nj)

        # if supernode k is not a root node, compute and push update matrix onto stack
        if na > 0:   
            # compute L_{Ak,Nk} := A_{Ak,Nk}*inv(L_{Nk,Nk}')
            blas.trsm(F, F, m = na, n = nn, ldA = nj, 
                      ldB = nj, offsetB = nn, transA = 'T', side = 'R')

            # compute Uk = Uk - L_{Ak,Nk}*inv(D_{Nk,Nk})*L_{Ak,Nk}'
            if nn == 1:
                blas.syr(F, F, n = na, offsetx = nn, \
                         offsetA = nn*nj+nn, ldA = nj, alpha = -1.0)
            else:
                blas.syrk(F, F, k = nn, n = na, offsetA = nn, ldA = nj,
                          offsetC = nn*nj+nn, ldC = nj, alpha = -1.0, beta = 1.0)

            # compute L_{Ak,Nk} := L_{Ak,Nk}*inv(L_{Nk,Nk})
            blas.trsm(F, F, m = na, n = nn,\
                      ldA = nj, ldB = nj, offsetB = nn, side = 'R')

            # add Uk to stack
            Uk = matrix(0.0,(na,na))
            lapack.lacpy(F, Uk, m = na, n = na, uplo = 'L', offsetA = nn*nj+nn, ldA = nj)
            stack.append(Uk)

        # copy the leading Nk columns of frontal matrix to blkval
        lapack.lacpy(F, blkval, uplo = "L", offsetB = blkptr[k], m = nj, n = nn, ldB = nj)        

    X.is_factor = True

    return

