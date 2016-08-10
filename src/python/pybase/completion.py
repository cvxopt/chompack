from cvxopt import matrix, blas, lapack
from chompack.symbolic import cspmatrix
from chompack.misc import frontal_get_update, frontal_get_update_factor

def completion(X, factored_updates = True):
    """
    Supernodal multifrontal maximum determinant positive definite
    matrix completion. The routine computes the Cholesky factor
    :math:`L` of the inverse of the maximum determinant positive
    definite matrix completion of :math:`X`:, i.e.,

    .. math::
         P( S^{-1} ) = X

    where :math:`S = LL^T`. On exit, the argument `X` contains the
    lower-triangular Cholesky factor :math:`L`.

    The optional argument `factored_updates` can be used to enable (if
    True) or disable (if False) updating of intermediate
    factorizations.

    :param X:                 :py:class:`cspmatrix`
    :param factored_updates:  boolean
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

    for k in reversed(list(snpost)):

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
            i = chidx[ii]       
            if factored_updates:
                r = relidx[relptr[i]:relptr[i+1]]
                stack.append(frontal_get_update_factor(F,r,nn,na))
            else:
                stack.append(frontal_get_update(F,relidx,relptr,i))

        # if supernode k is not a root node:
        if na > 0:
            if factored_updates:
                # In this case we have Vk = Lk'*Lk
                trL1 = 'T'
                trL2 = 'N'                
            else:
                # factorize Vk 
                lapack.potrf(F, offsetA = nj*nn+nn, n = na, ldA = nj)
                # In this case we have Vk = Lk*Lk'
                trL1 = 'N'  
                trL2 = 'T'  

            # compute L_{Ak,Nk} and inv(D_{Nk,Nk}) = S_{Nk,Nk} - S_{Ak,Nk}'*L_{Ak,Nk}
            lapack.trtrs(F, blkval, offsetA = nj*nn+nn, trans = trL1,\
                         offsetB = blkptr[k]+nn, ldB = nj, n = na, nrhs = nn)
            blas.syrk(blkval, blkval, n = nn, k = na, trans= 'T', alpha = -1.0, beta = 1.0,
                      offsetA = blkptr[k]+nn, offsetC = blkptr[k], ldA = nj, ldC = nj)
            lapack.trtrs(F, blkval, offsetA = nj*nn+nn, trans = trL2,\
                         offsetB = blkptr[k]+nn, ldB = nj, n = na, nrhs = nn)                
            for i in range(nn):
                blas.scal(-1.0, blkval, n = na, offset = blkptr[k] + i*nj + nn)


        # factorize inv(D_{Nk,Nk}) as R*R' so that D_{Nk,Nk} = L*L' with L = inv(R)'
        lapack.lacpy(blkval, F, offsetA = blkptr[k], ldA = nj,\
                     ldB = nj, m = nn, n = nn, uplo = 'L') # copy    -- FIX!
        F[:nn,:nn] = matrix(F[:nn,:nn][::-1],(nn,nn))      # reverse -- FIX!
        lapack.potrf(F, ldA = nj, n = nn, uplo = 'U')      # factorize
        F[:nn,:nn] = matrix(F[:nn,:nn][::-1],(nn,nn))      # reverse -- FIX!
        lapack.lacpy(F, blkval, offsetB = blkptr[k], ldA = nj,\
                     ldB = nj, m = nn, n = nn, uplo = 'L') # copy    -- FIX!

        # compute L = inv(R')
        lapack.trtri(blkval, offsetA = blkptr[k], ldA = nj, n = nn)

    X._is_factor = True

    return
