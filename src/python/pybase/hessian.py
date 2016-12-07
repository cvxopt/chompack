from cvxopt import matrix, blas, lapack
from chompack.symbolic import cspmatrix
from chompack.misc import frontal_get_update_factor, frontal_get_update, frontal_add_update, tril

def __Y2K(L, U, inv = False):

    n = L.symb.n
    snpost = L.symb.snpost
    snptr = L.symb.snptr
    chptr = L.symb.chptr
    chidx = L.symb.chidx

    relptr = L.symb.relptr
    relidx = L.symb.relidx
    blkptr = L.symb.blkptr

    stack = []

    alpha = 1.0
    if inv: alpha = -1.0

    for Ut in U:
        for k in snpost:

            nn = snptr[k+1]-snptr[k]       # |Nk|
            na = relptr[k+1]-relptr[k]     # |Ak|
            nj = na + nn

            # allocate F and copy Ut_{Jk,Nk} to leading columns of F
            F = matrix(0.0, (nj,nj))
            lapack.lacpy(Ut.blkval, F, offsetA = blkptr[k], ldA = nj, m = nj, n = nn, uplo = 'L')

            if not inv:
                # add update matrices from children to frontal matrix
                for i in range(chptr[k+1]-1,chptr[k]-1,-1):
                    Ui = stack.pop()
                    frontal_add_update(F, Ui, relidx, relptr, chidx[i])

            if na > 0:
                # F_{Ak,Ak} := F_{Ak,Ak} - alpha*L_{Ak,Nk}*F_{Ak,Nk}'
                blas.gemm(L.blkval, F, F, beta = 1.0, alpha = -alpha, m = na, n = na, k = nn,\
                          ldA = nj, ldB = nj, ldC = nj, transB = 'T',\
                          offsetA = blkptr[k]+nn, offsetB = nn, offsetC = nn*(nj+1))
                # F_{Ak,Nk} := F_{Ak,Nk} - alpha*L_{Ak,Nk}*F_{Nk,Nk}
                blas.symm(F, L.blkval, F, side = 'R', beta = 1.0, alpha = -alpha,\
                          m = na, n = nn, ldA = nj, ldB = nj, ldC = nj,\
                          offsetA = 0, offsetB = blkptr[k]+nn, offsetC = nn)
                # F_{Ak,Ak} := F_{Ak,Ak} - alpha*F_{Ak,Nk}*L_{Ak,Nk}'
                blas.gemm(F, L.blkval, F, beta = 1.0, alpha = -alpha, m = na, n = na, k = nn,\
                          ldA = nj, ldB = nj, ldC = nj, transB = 'T',\
                          offsetA = nn, offsetB = blkptr[k]+nn, offsetC = nn*(nj+1))

            if inv:
                # add update matrices from children to frontal matrix
                for i in range(chptr[k+1]-1,chptr[k]-1,-1):
                    Ui = stack.pop()
                    frontal_add_update(F, Ui, relidx, relptr, chidx[i])

            if na > 0:
                # add Uk' to stack
                Uk = matrix(0.0,(na,na))
                lapack.lacpy(F, Uk, m = na, n = na, offsetA = nn*(nj+1), ldA = nj, uplo = 'L')
                stack.append(Uk)

            # copy the leading Nk columns of frontal matrix to blkval
            lapack.lacpy(F, Ut.blkval, uplo = 'L', offsetB = blkptr[k], m = nj, n = nn, ldB = nj)

    return

def __M2T(L, U, inv = False):

    n = L.symb.n
    snpost = L.symb.snpost
    snptr = L.symb.snptr
    chptr = L.symb.chptr
    chidx = L.symb.chidx

    relptr = L.symb.relptr
    relidx = L.symb.relidx
    blkptr = L.symb.blkptr

    stack = []

    alpha = 1.0
    if inv: alpha = -1.0

    for Ut in U:
        for k in reversed(list(snpost)):

            nn = snptr[k+1]-snptr[k]       # |Nk|
            na = relptr[k+1]-relptr[k]     # |Ak|
            nj = na + nn

            # allocate F and copy Ut_{Jk,Nk} to leading columns of F
            F = matrix(0.0, (nj,nj))
            lapack.lacpy(Ut.blkval, F, offsetA = blkptr[k], ldA = nj, m = nj, n = nn, uplo = 'L')

            # if supernode k is not a root node:
            if na > 0:
                # copy Vk to 2,2 block of F
                Vk = stack.pop()
                lapack.lacpy(Vk, F, offsetB = nn*(nj+1), m = na, n = na, uplo = 'L')

            ## compute T_{Jk,Nk} (stored in leading columns of F)

            if inv:
                # if supernode k has any children:
                for ii in range(chptr[k],chptr[k+1]):
                    stack.append(frontal_get_update(F,relidx,relptr,chidx[ii]))

            # if supernode k is not a root node:
            if na > 0:
                # F_{Nk,Nk} := F_{Nk,Nk} - alpha*F_{Ak,Nk}'*L_{Ak,Nk}
                blas.gemm(F, L.blkval, F, beta = 1.0, alpha = -alpha, m = nn, n = nn, k = na,\
                          transA = 'T', ldA = nj, ldB = nj, ldC = nj,\
                          offsetA = nn, offsetB = blkptr[k]+nn, offsetC = 0)
                # F_{Ak,Nk} := F_{Ak,Nk} - alpha*F_{Ak,Ak}*L_{Ak,Nk}
                blas.symm(F, L.blkval, F, side = 'L', beta = 1.0, alpha = -alpha,\
                          m = na, n = nn, ldA = nj, ldB = nj, ldC = nj,\
                          offsetA = (nj+1)*nn, offsetB = blkptr[k]+nn, offsetC = nn)
                # F_{Nk,Nk} := F_{Nk,Nk} - alpha*L_{Ak,Nk}'*F_{Ak,Nk}
                blas.gemm(L.blkval, F, F, beta = 1.0, alpha = -alpha, m = nn, n = nn, k = na,\
                          transA = 'T', ldA = nj, ldB = nj, ldC = nj,\
                          offsetA = blkptr[k]+nn, offsetB = nn, offsetC = 0)

            # copy the leading Nk columns of frontal matrix to Ut
            lapack.lacpy(F, Ut.blkval, offsetB = blkptr[k], ldB = nj, m = nj, n = nn, uplo = 'L')

            if not inv:
                # if supernode k has any children:
                for ii in range(chptr[k],chptr[k+1]):
                    stack.append(frontal_get_update(F,relidx,relptr,chidx[ii]))

    return

def __scale(L, Y, U, adj = False, inv = False, factored_updates = True):

    n = L.symb.n
    snpost = L.symb.snpost
    snptr = L.symb.snptr
    chptr = L.symb.chptr
    chidx = L.symb.chidx

    relptr = L.symb.relptr
    relidx = L.symb.relidx
    blkptr = L.symb.blkptr

    stack = []

    for k in reversed(list(snpost)):

        nn = snptr[k+1]-snptr[k]       # |Nk|
        na = relptr[k+1]-relptr[k]     # |Ak|
        nj = na + nn

        F = matrix(0.0, (nj,nj))
        lapack.lacpy(Y.blkval, F, m = nj, n = nn, ldA = nj, offsetA = blkptr[k], uplo = 'L')

        # if supernode k is not a root node:
        if na > 0:
            # copy Vk to 2,2 block of F
            Vk = stack.pop()
            lapack.lacpy(Vk, F, ldB = nj, offsetB = nn*(nj+1), m = na, n = na, uplo = 'L')

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
                if adj is False: trns = 'N'
                elif adj is True: trns = 'T'
            else:
                # factorize Vk
                lapack.potrf(F, offsetA = nj*nn+nn, n = na, ldA = nj)
                # In this case we have Vk = Lk*Lk'
                if adj is False: trns = 'T'
                elif adj is True: trns = 'N'

        if adj is False: tr = ['T','N']
        elif adj is True: tr = ['N','T']

        if inv is False:
            for Ut in U:
                # symmetrize (1,1) block of Ut_{k} and scale
                U11 = matrix(0.0,(nn,nn))
                lapack.lacpy(Ut.blkval,U11, offsetA = blkptr[k], m = nn, n = nn, ldA = nj, uplo = 'L')
                U11 += U11.T
                U11[::nn+1] *= 0.5
                lapack.lacpy(U11,Ut.blkval, offsetB = blkptr[k], m = nn, n = nn, ldB = nj, uplo = 'N')

                blas.trsm(L.blkval, Ut.blkval, side = 'R', transA = tr[0],\
                          m = nj, n = nn, offsetA = blkptr[k], ldA = nj,\
                          offsetB = blkptr[k], ldB = nj)
                blas.trsm(L.blkval, Ut.blkval, m = nn, n = nn, transA = tr[1],\
                          offsetA = blkptr[k], offsetB = blkptr[k],\
                          ldA = nj, ldB = nj)

                # zero-out strict upper triangular part of {Nj,Nj} block 
                for i in range(1,nn): blas.scal(0.0, Ut.blkval, offset = blkptr[k] + nj*i, n = i) 
                
                if na > 0: blas.trmm(F, Ut.blkval, m = na, n = nn, transA = trns,\
                                     offsetA = nj*nn+nn, ldA = nj,\
                                     offsetB = blkptr[k]+nn, ldB = nj)
        else: # inv is True
            for Ut in U:
                # symmetrize (1,1) block of Ut_{k} and scale
                U11 = matrix(0.0,(nn,nn))
                lapack.lacpy(Ut.blkval,U11, offsetA = blkptr[k], m = nn, n = nn, ldA = nj, uplo = 'L')
                U11 += U11.T
                U11[::nn+1] *= 0.5
                lapack.lacpy(U11,Ut.blkval, offsetB = blkptr[k], m = nn, n = nn, ldB = nj, uplo = 'N')

                blas.trmm(L.blkval, Ut.blkval, side = 'R', transA = tr[0],\
                          m = nj, n = nn, offsetA = blkptr[k], ldA = nj,\
                          offsetB = blkptr[k], ldB = nj)
                blas.trmm(L.blkval, Ut.blkval, m = nn, n = nn, transA = tr[1],\
                          offsetA = blkptr[k], offsetB = blkptr[k],\
                          ldA = nj, ldB = nj)

                # zero-out strict upper triangular part of {Nj,Nj} block 
                for i in range(1,nn): blas.scal(0.0, Ut.blkval, offset = blkptr[k] + nj*i, n = i) 
                
                if na > 0: blas.trsm(F, Ut.blkval, m = na, n = nn, transA = trns,\
                                     offsetA = nj*nn+nn, ldA = nj,\
                                     offsetB = blkptr[k]+nn, ldB = nj)

    return

def hessian(L, Y, U, adj = False, inv = False, factored_updates = False):
    """
    Supernodal multifrontal Hessian mapping.

    The mapping

    .. math::
         \mathcal H_X(U) = P(X^{-1}UX^{-1})

    is the Hessian of the log-det barrier at a positive definite chordal
    matrix :math:`X`, applied to a symmetric chordal matrix :math:`U`. The Hessian operator
    can be factored as

    .. math::
         \mathcal H_X(U) = \mathcal G_X^{\mathrm adj}( \mathcal G_X(U) )

    where the mappings on the right-hand side are adjoint mappings
    that map chordal symmetric matrices to chordal symmetric matrices.

    This routine evaluates the mapping :math:`G_X` and its adjoint
    :math:`G_X^{\mathrm adj}` as well as the corresponding inverse
    mappings. The inputs `adj` and `inv` control the action as
    follows:

    +--------------------------------------------------+--------+-------+
    | Action                                           |`adj`   | `inv` |
    +==================================================+========+=======+
    | :math:`U = \mathcal G_X(U)`                      | False  | False |
    +--------------------------------------------------+--------+-------+
    | :math:`U = \mathcal G_X^{\mathrm adj}(U)`        | True   | False |
    +--------------------------------------------------+--------+-------+
    | :math:`U = \mathcal G_X^{-1}(U)`                 | False  | True  |
    +--------------------------------------------------+--------+-------+
    | :math:`U = \mathcal (G_X^{\mathrm adj})^{-1}(U)` | True   | True  |
    +--------------------------------------------------+--------+-------+

    The input argument :math:`L` is the Cholesky factor of
    :math:`X`. The input argument :math:`Y` is the projected inverse of
    :math:`X`. The input argument :math:`U` is either a chordal matrix (a
    :py:class:`cspmatrix`) of a list of chordal matrices with the same
    sparsity pattern as :math:`L` and :math:`Y`.

    The optional argument `factored_updates` can be used to enable (if
    True) or disable (if False) updating of intermediate
    factorizations.

    :param L:                 :py:class:`cspmatrix` (factor)
    :param Y:                 :py:class:`cspmatrix`
    :param U:                 :py:class:`cspmatrix` or list of :py:class:`cspmatrix` objects
    :param adj:               boolean
    :param inv:               boolean
    :param factored_updates:  boolean
    """
    assert L.symb == Y.symb, "Symbolic factorization mismatch"
    assert isinstance(L, cspmatrix) and L.is_factor is True, "L must be a cspmatrix factor"
    assert isinstance(Y, cspmatrix) and Y.is_factor is False, "Y must be a cspmatrix"

    if isinstance(U, cspmatrix):
        assert U.is_factor is False,\
            "U must be a cspmatrix or a list of cbsmatrices"
        U = [U]
    else:
        for Ut in U:
            assert Ut.symb == L.symb, "Symbolic factorization mismatch"
            assert isinstance(Ut, cspmatrix) and Ut.is_factor is False,\
                "U must be a cspmatrix or a list of cbsmatrices"

    if adj is False and inv is False:
        __Y2K(L, U, inv = inv)
        __scale(L, Y, U, inv = inv, adj = adj, factored_updates = factored_updates)

    elif adj is True and inv is False:
        __scale(L, Y, U, inv = inv, adj = adj, factored_updates = factored_updates)
        __M2T(L, U, inv = inv)

    elif adj is True and inv is True:
        __M2T(L, U, inv = inv)
        __scale(L, Y, U, inv = inv, adj = adj, factored_updates = factored_updates)

    elif adj is False and inv is True:
        __scale(L, Y, U, inv = inv, adj = adj, factored_updates = factored_updates)
        __Y2K(L, U, inv = inv)

    elif adj is None and inv is False:
        __Y2K(L, U, inv = inv)
        __scale(L, Y, U, inv = inv, adj = False, factored_updates = factored_updates)
        __scale(L, Y, U, inv = inv, adj = True,  factored_updates = factored_updates)
        __M2T(L, U, inv = inv)

    elif adj is None and inv is True:
        __M2T(L, U, inv = inv)
        __scale(L, Y, U, inv = inv, adj = True,  factored_updates = factored_updates)
        __scale(L, Y, U, inv = inv, adj = False, factored_updates = factored_updates)
        __Y2K(L, U, inv = inv)

    return
