from cvxopt import matrix, spmatrix
from chompack.misc import symmetrize

def maxcardsearch(A, ve = None):
    """
    Maximum cardinality search ordering of a sparse chordal matrix.

    Returns the maximum cardinality search ordering of a symmetric
    chordal matrix :math:`A`. Only the lower triangular part of
    :math:`A` is accessed. The maximum cardinality search ordering
    is a perfect elimination ordering in the factorization
    :math:`PAP^T = LL^T`. The optional argument `ve` is the index of
    the last vertex to be eliminated (the default value is n-1).

    :param A:   :py:class:`spmatrix`

    :param ve:  integer between 0 and `A.size[0]`-1 (optional)
    
    """
    
    n = A.size[0]
    assert A.size[1] == n, "A must be a square matrix"
    assert type(A) is spmatrix, "A must be a sparse matrix"
    if ve is None: 
        ve = n-1
    else:
        assert type(ve) is int and 0<=ve<n,\
          "ve must be an integer between 0 and A.size[0]-1"    
    As = symmetrize(A)
    cp,ri,_ = As.CCS
    
    # permutation vector 
    p = matrix(0,(n,1))
    
    # weight array
    w = matrix(0,(n,1))
    max_w = 0
    S = [list(range(ve))+list(range(ve+1,n))+[ve]] + [[] for i in range(n-1)]
        
    for i in range(n-1,-1,-1):
        while True:
            if len(S[max_w]) > 0:
                v = S[max_w].pop()
                if w[v] >= 0: break
            else:
                max_w -= 1
        p[i] = v    
        w[v] = -1   # set w[v] = -1 to mark that node v has been numbered

        # increase weights for all unnumbered neighbors
        for r in ri[cp[v]:cp[v+1]]:
            if w[r] >= 0: 
                w[r] += 1
                S[w[r]].append(r)     # bump r up to S[w[r]]
                max_w = max(max_w,w[r])   

    return p
