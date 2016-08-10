from cvxopt import spmatrix, sparse, matrix, blas, lapack
__all__ = []

def tril(A):
    """
    Returns the lower triangular part of :math:`A`.
    """
    if isinstance(A,spmatrix):
        idx = [i for i,ij in enumerate(zip(A.I,A.J)) if ij[0]>=ij[1]]
        return spmatrix(A.V[idx],A.I[idx],A.J[idx],A.size)
    elif isinstance(A,matrix):
        B = matrix(0.0, A.size)
        lapack.lacpy(A, B, uplo = 'L')
        return B
    else:
        raise TypeError

def triu(A):
    """
    Returns the upper triangular part of :math:`A`.
    """
    if isinstance(A,spmatrix):
        idx = [i for i,ij in enumerate(zip(A.I,A.J)) if ij[0]<=ij[1]]
        return spmatrix(A.V[idx],A.I[idx],A.J[idx],A.size)
    elif isinstance(A,matrix):
        B = matrix(0.0, A.size)
        lapack.lacpy(A, B, uplo = 'U')
        return B
    else:
        raise TypeError

def symmetrize(A):
    """
    Returns a symmetric matrix from a sparse square matrix :math:`A`. Only the
    lower triangular entries of :math:`A` are accessed.
    """
    assert type(A) is spmatrix, "argument must be a sparse matrix"
    assert A.size[0] == A.size[1], "argument must me a square matrix"
    idx = [i for i,ij in enumerate(zip(A.I,A.J)) if ij[0] > ij[1]]
    return tril(A) + spmatrix(A.V[idx], A.J[idx], A.I[idx], A.size)

def perm(A, p):
    """
    Symmetric permutation of a symmetric sparse matrix. 

    :param A:    :py:class:`spmatrix`
    :param p:    :py:class:`matrix` or :class:`list` of length `A.size[0]`

    """
    assert isinstance(A,spmatrix), "argument must be a sparse matrix"
    assert A.size[0] == A.size[1], "A must be a square matrix"
    assert A.size[0] == len(p), "length of p must be equal to the order of A"
    return A[p,p]

def eye(n):
    """
    Identity matrix of order n.
    """
    I = matrix(0.0,(n,n))
    I[::n+1] = 1.0
    return I

try:
    from chompack.cbase import frontal_add_update
except:
    def frontal_add_update(F, U, relidx, relptr, i, alpha = 1.0):
        """
        Add update matrix to frontal matrix.
        """
        r = relidx[relptr[i]:relptr[i+1]]
        F[r,r] += alpha*U
        return 

try:
    from chompack.cbase import frontal_get_update
except:
    def frontal_get_update(F, relidx, relptr, i):
        """
        Extract update matrix from frontal matrix.
        """
        r = relidx[relptr[i]:relptr[i+1]]
        return F[r,r]

def frontal_get_update_factor(F, r, nn, na):
    """
    Computes factorization Vi = Li'*Li, given F

       F = [ S_{N_k,N_k}   S_{Ak,Nk}'  ]
           [ S_{Ak,Nk}'     Lk'*Lk     ]

    where Vk = Lk'*Lk. Note that the 2,2 block of the argument 'F'
    stores Lk and not Lk'*Lk. The argument 'r' is a vector of relative
    indices r = [ i1, i2, ..., iN ], and the arguments 'nn' and 'na'
    are equal to |Nk| and |Ak|, respectively.

    Partition r as r = [r1;r2] such that indices in r1 are less than
    |Nk| and indices in r2 are greater than or equal to |Nk|. Then

       Vi = F[r,r] = [  A   B'  ] 
                     [  B  C'*C ]  

                   = [ Li_{11}'  Li_{21}' ] [ Li_{11}     0    ]
                     [    0      Li_{22}' ] [ Li_{21}  Li_{22} ]

                   = Li'*Li
    where

       A = F[r1,r1],   B = F[r2,r1],   C = F[nn:,r2].

    The matrix L_{22} is obtained from C by means of a series of
    Householder transformations, Li_{21} can be computed by solving
    Li_{22}'*Li_{21} = B, and Li_{11} can be computed by factoring the
    matrix A - Li_{21}'*Li_{21} = Li_{11}'*Li_{11}.

    ARGUMENTS
    F         square matrix 

    r         vector of indices

    nn        integer

    na        integer

    RETURNS
    Li        matrix with lower triangular Cholesky factor
    """

    N = len(r)

    # allocate storage for lower triangular factor
    Li = matrix(0.0,(N,N))

    # Compute:
    #   N1 = number of entries in r such that r[i] < nn
    #   N2 = number of entries in r such that r[i] >= nn
    N1 = 0
    for i in range(len(r)):
        if r[i] >= nn: break
        else: N1 += 1
    N2 = N-N1

    # extract A and B from F; copy to leading N1 columns of Li
    Li[:,:N1] = F[r,r[:N1]]

    if N2 > 0:
        # extract C and reduce C to lower triangular from
        C = F[nn:,r[N1:]]
        for ii in range(N2):
            col = r[N-1-ii] - nn       # index of column in C
            t = (na - col) - (ii + 1)  # number of nonzeros to zero out
            if t == 0: continue        

            # compute and apply Householder reflector
            tau = lapack.larfg(C, C, n = t+1, offseta = na*(N2-ii)-1-ii, offsetx = na*(N2-ii)-1-ii-t)
            tmp = C[na*(N2-ii)-1-ii]
            C[na*(N2-ii)-1-ii] = 1.0
            lapack.larfx(C, tau, C, m = t+1, n = N2-1-ii, ldC = na, offsetv = na*(N2-ii)-1-ii-t, offsetC = na-1-ii-t)
            C[na*(N2-ii)-1-ii] = tmp

        # copy lower triangular matrix from C to 2,2 block of Li
        lapack.lacpy(C, Li, uplo = 'L', m = N2, n = N2, offsetA = na-N2, ldA = na, offsetB = (N+1)*N1, ldB = N)

        # compute Li_{21} by solving L_{22}'*L_{21} = B
        lapack.trtrs(Li, Li, trans = 'T', n = N2, nrhs = N1,\
                     ldA = N, ldB = N, offsetB = N1, offsetA = N1*N+N1)

        # compute A - Li_{21}'*Li_{21}
        blas.syrk(Li, Li, trans = 'T', n = N1, k = N2,\
                  ldA = N, ldC = N, offsetA = N1, alpha = -1.0, beta = 1.0)

    # compute Li_{11}  [reverse -- factorize -- reverse]
    At = matrix(Li[:N1,:N1][::-1],(N1,N1)) 
    lapack.potrf(At, uplo = 'U')
    Li[:N1,:N1] = matrix(At[::-1],(N1,N1))
    
    return Li

try:
    from chompack.cbase import lmerge
except:
    def lmerge(left, right, offsetl = 0, offsetr = 0, nl = None, nr = None):
        if nl is None: nl = len(left)
        if nr is None: nr = len(right)
        tmp = matrix(0,(nl+nr,1))
        il = 0; ir = 0; k = 0
        while il < nl and ir < nr:
            if left[offsetl+il] < right[offsetr+ir]:
                tmp[k] = left[offsetl+il]
                il += 1
            elif left[offsetl+il] > right[offsetr+ir]:
                tmp[k] = right[offsetr+ir]
                ir += 1
            else:
                tmp[k] = left[offsetl+il]
                il += 1; ir += 1
            k += 1
        if il < nl:
            tmp[k:k+(nl-il)] = left[offsetl+il:offsetl+nl]
            k += nl-il
        if ir < nr:
            tmp[k:k+(nr-ir)] = right[offsetr+ir:offsetr+nr]
            k += nr-ir
        left[offsetl:offsetl+k] = tmp[:k]
        return k
