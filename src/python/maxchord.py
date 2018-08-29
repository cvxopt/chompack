from cvxopt import matrix, spmatrix
from chompack.misc import symmetrize
from itertools import chain

def maxchord(A, ve = None):
    """
    Maximal chordal subgraph of sparsity graph.

    Returns a lower triangular sparse matrix which is the projection
    of :math:`A` on a maximal chordal subgraph and a perfect
    elimination order :math:`p`. Only the
    lower triangular part of :math:`A` is accessed. The
    optional argument `ve` is the index of the last vertex to be
    eliminated (the default value is `n-1`). If :math:`A` is chordal,
    then the matrix returned is equal to :math:`A`.

    :param A:   :py:class:`spmatrix`

    :param ve:  integer between 0 and `A.size[0]`-1 (optional)

    .. seealso::

         P. M. Dearing, D. R. Shier, D. D. Warner, `Maximal chordal
         subgraphs <http://dx.doi.org/10.1016/0166-218X(88)90075-3>`_,
         Discrete Applied Mathematics, 20:3, 1988, pp. 181-190.

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
    cp,ri,val = As.CCS

    # permutation vector
    p = matrix(0,(n,1))

    # weight array
    w = matrix(0,(n,1))
    max_w = 0
    S = [list(range(ve))+list(range(ve+1,n))+[ve]] + [[] for i in range(n-1)]

    C = [set() for i in range(n)]
    E = [[] for i in range(n)]     # edge list
    V = [[] for i in range(n)]     # num. values

    for i in range(n-1,-1,-1):
        # select next node to number
        while True:
            if len(S[max_w]) > 0:
                v = S[max_w].pop()
                if w[v] >= 0: break
            else:
                max_w -= 1
        p[i] = v
        w[v] = -1   # set w[v] = -1 to mark that node v has been numbered

        # loop over unnumbered neighbors of node v
        for ii in range(cp[v],cp[v+1]):
            u = ri[ii]
            d = val[ii]
            if w[u] >= 0:
                if C[u].issubset(C[v]):
                    C[u].update([v])
                    w[u] += 1
                    S[w[u]].append(u)    # bump up u to S[w[u]]
                    max_w = max(max_w,w[u])  # update max deg.
                    E[min(u,v)].append(max(u,v))
                    V[min(u,v)].append(d)
            elif u == v:
                E[u].append(u)
                V[u].append(d)

    # build adjacency matrix of reordered max. chordal subgraph
    Am = spmatrix([d for d in chain.from_iterable(V)],[i for i in chain.from_iterable(E)],\
                  [i for i in chain.from_iterable([len(Ej)*[j] for j,Ej in enumerate(E)])],(n,n))

    return Am,p
