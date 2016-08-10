from cvxopt import matrix, blas, lapack
from chompack.symbolic import cspmatrix

try:
    from chompack.cbase import dot
except:
    def dot(X,Y):
        """
        Computes trace product of X and Y.
        """
        assert Y.symb == X.symb, "Symbolic factorization mismatch"
        assert X.is_factor is False, "cspmatrix factor object"
        assert Y.is_factor is False, "cspmatrix factor object"

        snptr = X.symb.snptr
        sncolptr = X.symb.sncolptr
        blkptr = X.symb.blkptr
        val = 0.0
        for k in range(X.symb.Nsn):
            offset = blkptr[k]
            nk = snptr[k+1]-snptr[k]
            ck = sncolptr[k+1]-sncolptr[k]
            for j in range(nk):
                val -= X.blkval[offset+ck*j+j]*Y.blkval[offset+ck*j+j]
                val += 2.0*blas.dot(X.blkval,Y.blkval,offsetx=offset+ck*j+j,offsety=offset+ck*j+j,n=ck-j)
        return val

def syr2(X, u, v, alpha = 1.0, beta = 1.0, reordered=False):
    r"""
    Computes the projected rank 2 update of a cspmatrix X

    .. math::
         X := \alpha*P(u v^T + v u^T) + \beta X.
    """
    assert X.is_factor is False, "cspmatrix factor object"
    symb = X.symb
    n = symb.n
    snptr = symb.snptr
    snode = symb.snode
    blkval = X.blkval
    blkptr = symb.blkptr
    relptr = symb.relptr
    snrowidx = symb.snrowidx
    sncolptr = symb.sncolptr

    if symb.p is not None and reordered is False:
        up = u[symb.p]
        vp = v[symb.p]
    else:
        up = u
        vp = v
            
    for k in range(symb.Nsn):
        nn = snptr[k+1]-snptr[k]     
        na = relptr[k+1]-relptr[k] 
        nj = na + nn

        for i in range(nn): blas.scal(beta, blkval, n = nj-i, offset = blkptr[k]+(nj+1)*i)

        uk = up[snrowidx[sncolptr[k]:sncolptr[k+1]]]
        vk = vp[snrowidx[sncolptr[k]:sncolptr[k+1]]]

        blas.syr2(uk, vk, blkval, n = nn, offsetA = blkptr[k], ldA = nj, alpha = alpha)
        blas.ger(uk, vk, blkval, m = na, n = nn, offsetx = nn, offsetA = blkptr[k]+nn, ldA = nj, alpha = alpha)
        blas.ger(vk, uk, blkval, m = na, n = nn, offsetx = nn, offsetA = blkptr[k]+nn, ldA = nj, alpha = alpha)
        
    return
