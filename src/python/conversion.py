from cvxopt import matrix, spdiag, spmatrix, sparse, amd
import itertools
from chompack.symbolic import symbolic
from chompack.misc import tril, symmetrize, perm

def symb_to_block(symb, coupling = 'full'):
    """
    Maps a symbolic factorization to a block-diagonal structure with
    coupling constraints.

    :param symb:               :py:class:`symbolic`
    :param coupling:           optional 
    :return dims:              list of block dimensions
    :return sparse_to_block:   dictionary
    :return constraints:       list of coupling constraints
    """

    n = len(symb.snode)             # order of block
    Ncliques = len(symb.snpar)      # number of cliques

    # compute clique orders
    dims = [symb.sncolptr[j+1]-symb.sncolptr[j] for j in range(Ncliques)]

    # compute offsets in block-diagonal structure
    offsets = [0]
    for i in range(Ncliques): offsets.append(offsets[-1] + dims[i]**2)

    constraints = []        # list of coupling constraints
    sparse_to_block = {}    # conversion dictionary

    for k in range(Ncliques):
        # map nonzeros in {Jk,Nk} part of clique k to block-diagonal structure
        nodes = symb.snode[symb.snptr[k]:symb.snptr[k+1]]
        rows = symb.snrowidx[symb.sncolptr[k]:symb.sncolptr[k+1]]
        nk = len(nodes)    # number of nodes in supernode
        wk = len(rows)     # number of nodes in clique
        for j in range(nk):
            for i in range(j,wk):
                if i == j:
                    sparse_to_block[nodes[j]*n + rows[i]] = (offsets[k] + j*wk + i,)
                else:
                    sparse_to_block[nodes[j]*n + rows[i]] =(offsets[k] + j*wk + i, offsets[k] + i*wk + j)

        # add coupling constraints to list of constraints
        if symb.snpar[k] == k: continue   # skip if supernode k is a root supernode
        p = symb.snpar[k]
        np = len(symb.snode[symb.snptr[p]:symb.snptr[p+1]])
        wp = symb.sncolptr[p+1] - symb.sncolptr[p]
        ri = symb.relidx[symb.relptr[k]:symb.relptr[k+1]]

        if type(coupling) is spmatrix:           
            tmp = coupling[rows[nk:],rows[nk:]]
            for i,j in zip(tmp.I,tmp.J):
                if j == i:
                    constraints.append((offsets[k] + (j+nk)*wk + i+nk,
                                        offsets[p] + ri[j]*wp + ri[i]))
                else:
                    constraints.append((offsets[k] + (j+nk)*wk + i+nk,
                                        offsets[p] + ri[j]*wp + ri[i],
                                        offsets[k] + (i+nk)*wk + j+nk,
                                        offsets[p] + ri[i]*wp + ri[j]))   
                     
        elif coupling == 'full':
            for j in range(len(ri)):
                for i in range(j,len(ri)):
                    if j == i:
                        constraints.append((offsets[k] + (j+nk)*wk + i+nk,
                                            offsets[p] + ri[j]*wp + ri[i]))
                    else:
                        constraints.append((offsets[k] + (j+nk)*wk + i+nk,
                                            offsets[p] + ri[j]*wp + ri[i],
                                            offsets[k] + (i+nk)*wk + j+nk,
                                            offsets[p] + ri[i]*wp + ri[j]))

    return dims, sparse_to_block, constraints


def convert_block(G, h, dim, **kwargs):
    r"""
    Applies the clique conversion method to a single positive
    semidefinite block of a cone linear program

    .. math::
        \begin{array}{ll}
           \mbox{maximize}   & -h^T z \\
           \mbox{subject to} &  G^T z + c = 0 \\
                             &  \mathbf{smat}(z)\ \ \text{psd completable}
        \end{array}

    After conversion, the above problem is converted to a block-diagonal one 

    .. math::
        \begin{array}{ll}
           \mbox{maximize}   & -h_b^T z_b  \\
           \mbox{subject to} &  G_b^T z_b + c = 0 \\
                             &  G_c^T z_b = 0 \\
                             &  \mathbf{smat}(z_b)\ \ \text{psd block-diagonal}
        \end{array}
                             
    where :math:`z_b` is a vector representation of a block-diagonal
    matrix. The constraint :math:`G_b^T z_b + c = 0` corresponds to
    the original constraint :math:`G'z + c = 0`, and the constraint
    :math:`G_c^T z_b = 0` is a coupling constraint.
       
    :param G:                 :py:class:`spmatrix`
    :param h:                 :py:class:`matrix`
    :param dim:               integer
    :param merge_function:    routine that implements a merge heuristic (optional)
    :param coupling:          mode of conversion (optional)
    :param max_density:       float (default: 0.4)

    The following example illustrates how to apply the conversion method to a one-block SDP:
    
    .. code-block:: python

        block = (G, h, dim) 
        blockc, blk2sparse, symb = convert_block(*block)
    
    The return value `blk2sparse` is a 4-tuple
    (`blki,I,J,n`) that defines a mapping between the sparse
    matrix representation and the converted block-diagonal
    representation. If `blkvec` represents a block-diagonal matrix,
    then

    .. code-block:: python

        S = spmatrix(blkvec[blki], I, J) 

    maps `blkvec` into is a sparse matrix representation of the
    matrix. Similarly, a sparse matrix `S` can be converted to the
    block-diagonal matrix representation using the code
    
    .. code-block:: python

        blkvec = matrix(0.0, (len(S),1), tc=S.typecode)
        blkvec[blki] = S.V

    The optional argument `max_density` controls whether or not to perform
    conversion based on the aggregate sparsity of the block. Specifically,
    conversion is performed whenever the number of lower triangular nonzeros
    in the aggregate sparsity pattern is less than or equal to `max_density*dim`.
        
    The optional argument `coupling` controls the introduction
    of equality constraints in the conversion. Possible values
    are *full* (default), *sparse*, *sparse+tri*, and any nonnegative
    integer. Full coupling results in a conversion in which all
    coupling constraints are kept, and hence the converted problem is
    equivalent to the original problem. Sparse coupling yeilds a
    conversion in which only the coupling constraints corresponding to
    nonzero entries in the aggregate sparsity pattern are kept, and
    sparse-plus-tridiagonal (*sparse+tri*) yeilds a conversion with
    tridiagonal coupling in addition to coupling constraints corresponding
    to nonzero entries in the aggregate sparsity pattern. Setting `coupling`
    to a nonnegative integer *k* yields a conversion with coupling
    constraints corresponding to entries in a band with half-bandwidth *k*.

    .. seealso::

        M. S. Andersen, A. Hansson, and L. Vandenberghe, `Reduced-Complexity
        Semidefinite Relaxations of Optimal Power Flow Problems
        <http://dx.doi.org/10.1109/TPWRS.2013.2294479>`_,
        IEEE Transactions on Power Systems, 2014.
        
    """
    
    merge_function = kwargs.get('merge_function', None)
    coupling = kwargs.get('coupling', 'full')
    tskip = kwargs.get('max_density',0.4)
    
    tc = G.typecode
    
    ###
    ### Find filled pattern, compute symbolic factorization using AMD
    ### ordering, and do "symbolic conversion"
    ###
    
    # find aggregate sparsity pattern
    h = sparse(h)
    LIa = matrix(list(set(G.I).union(set(h.I))))
    Ia = [i%dim for i in LIa]
    Ja = [j//dim for j in LIa]
    Va = spmatrix(1.,Ia,Ja,(dim,dim))
    
    # find permutation, symmetrize, and permute
    Va = symmetrize(tril(Va))
        
    # if not very sparse, skip decomposition 
    if float(len(Va))/Va.size[0]**2 > tskip:
        return (G, h, None, [dim]), None, None
    
    # compute symbolic factorization 
    F = symbolic(Va, merge_function = merge_function, p = amd.order)
    p = F.p
    ip = F.ip
    Va = F.sparsity_pattern(reordered = True, symmetric = True)
    
    # symbolic conversion
    if coupling == 'sparse': coupling = tril(Va)
    elif coupling == 'sparse+tri': 
        coupling = tril(Va)
        coupling += spmatrix(1.0,[i for j in range(Va.size[0]) for i in range(j,min(Va.size[0],j+2))],\
                             [j for j in range(Va.size[0]) for i in range(j,min(Va.size[0],j+2))],Va.size)
    elif type(coupling) is int:
        assert coupling >= 0
        bw = +coupling
        coupling = spmatrix(1.0,[i for j in range(Va.size[0]) for i in range(j,min(Va.size[0],j+bw+1))],\
                            [j for j in range(Va.size[0]) for i in range(j,min(Va.size[0],j+bw+1))],Va.size)
        
    dims, sparse_to_block, constraints = symb_to_block(F, coupling = coupling)
        
    # dimension of block-diagonal representation
    N = sum([d**2 for d in dims])      

    ###
    ### Convert problem data 
    ###
    
    m = G.size[1]           # cols in G
    cp, ri, val = G.CCS   
    
    IV = []                 # list of m (row, value) tuples
    J = []
    for j in range(m):
        iv = []
        for i in range(cp[j+1]-cp[j]):
            row = ri[cp[j]+i]%dim
            col = ri[cp[j]+i]//dim
            if row < col: continue   # ignore upper triangular entries
            k1 = ip[row]
            k2 = ip[col]
            blk_idx = sparse_to_block[min(k1,k2)*dim + max(k1,k2)]
            if k1 == k2:
                iv.append((blk_idx[0], val[cp[j]+i]))
            elif k1 > k2:
                iv.append((blk_idx[0], val[cp[j]+i]))
                iv.append((blk_idx[1], val[cp[j]+i].conjugate()))
            else:
                iv.append((blk_idx[0], val[cp[j]+i].conjugate()))
                iv.append((blk_idx[1], val[cp[j]+i]))                    
        iv.sort(key=lambda x: x[0])
        IV.extend(iv)
        J.extend(len(iv)*[j])
                    
    # build G_converted
    I, V = zip(*IV)
    G_converted = spmatrix(V, I, J, (N, m), tc = tc)
        
    # convert and build new h
    _, ri, val = h.CCS
    iv = []
    for i in range(len(ri)):
        row = ri[i]%dim
        col = ri[i]//dim
        if row < col: continue   # ignore upper triangular entries
        k1 = ip[row]
        k2 = ip[col]
        blk_idx = sparse_to_block[min(k1,k2)*dim + max(k1,k2)]
        if k1 == k2:
            iv.append((blk_idx[0], val[i]))
        elif k1 > k2:
            iv.append((blk_idx[0], val[i]))
            iv.append((blk_idx[1], val[i].conjugate()))
        else:
            iv.append((blk_idx[0], val[i].conjugate()))
            iv.append((blk_idx[1], val[i]))
    
    iv.sort(key=lambda x: x[0])
    if iv:
        I, V = zip(*iv)
    else:
        I, V = [], []
    h_converted = spmatrix(V, I, len(I)*[0], (N, 1), tc = tc)
    
    ###
    ### Build matrix representation of coupling constraints
    ###
    
    IV = []   # list of (row, value) tuples
    J = []
    ncon = 0
    for j in range(len(constraints)):
        iv = []
        if len(constraints[j]) == 2:
            ii, jj = constraints[j]
            iv = sorted([(ii, 1.0), (jj, -1.0)],key=lambda x: x[0])
            jl = 2*[ncon]
            ncon += 1
        elif len(constraints[j]) == 4:
            i1,j1,i2,j2 = constraints[j]
            iv = sorted([(i1, 1.0), (i2, 1.0), (j1, -1.0), (j2, -1.0)],key=lambda x: x[0])
            jl = 4*[ncon]
            ncon += 1
            if tc == 'z':
                iv.extend(sorted([(i1, complex(0.0,1.0)), (i2, complex(0.0,-1.0)),
                           (j1, complex(0.0,-1.0)), (j2, complex(0.0,1.0))],key=lambda x: x[0]))
                jl.extend(4*[ncon])
                ncon += 1
        IV.extend(iv)
        J.extend(jl)
                
    # build G_converted
    if IV: I, V = zip(*IV)
    else: I, V = [], []
    G_coupling = spmatrix(V, I, J, (N, ncon), tc = tc)
            
    # generate indices for reverse mapping (block_to_sparse)
    idx = []
    for k in sparse_to_block.keys():
        k1 = p[k%dim]
        k2 = p[k//dim]
        idx.append((min(k1,k2)*dim + max(k1,k2), sparse_to_block[k][0]))

    idx.sort()
    idx, blki = zip(*idx)
    blki = matrix(blki)
    I = [v%dim for v in idx]
    J = [v//dim for v in idx]
    n = sum([di**2 for di in dims])
    
    return (G_converted, h_converted, G_coupling, dims), (blki, I, J, n), F


def convert_conelp(c, G, h, dims, A = None, b = None, **kwargs):
    """
    Applies the clique conversion method of Fukuda et al. to the positive semidefinite blocks of a cone LP.

    :param c:                 :py:class:`matrix`
    :param G:                 :py:class:`spmatrix`
    :param h:                 :py:class:`matrix`
    :param dims:              dictionary
    :param A:                 :py:class:`spmatrix` or :py:class:`matrix`
    :param b:                 :py:class:`matrix`

    The following example illustrates how to convert a cone LP:
    
    .. code-block:: python

        prob = (c,G,h,dims,A,b)
        probc, blk2sparse, symbs = convert_conelp(*prob)
    
    The return value `blk2sparse` is a list of 4-tuples
    (`blki,I,J,n`) that each defines a mapping between the sparse
    matrix representation and the converted block-diagonal
    representation, and `symbs` is a list of symbolic factorizations
    corresponding to each of the semidefinite blocks in the original cone LP.
    
    .. seealso::

         M. Fukuda, M. Kojima, K. Murota, and K. Nakata, `Exploiting Sparsity
         in Semidefinite Programming via Matrix Completion I: General Framework
         <http://dx.doi.org/10.1137/S1052623400366218>`_,
         SIAM Journal on Optimization, 11:3, 2001, pp. 647-674.

         S. Kim, M. Kojima, M. Mevissen, and M. Yamashita, `Exploiting Sparsity
         in Linear and Nonlinear Matrix Inequalities via Positive Semidefinite
         Matrix Completion <http://dx.doi.org/10.1007/s10107-010-0402-6>`_,
         Mathematical Programming, 129:1, 2011, pp.. 33-68.

    """

    # extract linear and socp constraints
    offsets = dims['l'] + sum(dims['q'])
    G_lq = G[:offsets,:]
    h_lq = h[:offsets,0]

    # extract semidefinite blocks
    G_s = G[offsets:,:]
    h_s = h[offsets:,0]

    G_converted = [G_lq]; h_converted = [h_lq]
    G_coupling = []
    dims_list = []
    symbs = []
    
    offset = 0
    block_to_sparse = []
    for k, si in enumerate(dims['s']):
        # extract block
        G_b = G_s[offset:offset+si**2,:]
        h_b = h_s[offset:offset+si**2,0]
        offset += si**2

        # convert block
        blkk, b2s, F = convert_block(G_b, h_b, si, **kwargs)
        G1, h1, G2, blkdims = blkk
        G_converted.append(G1)
        h_converted.append(h1)
        dims_list.extend(blkdims)
        block_to_sparse.append(b2s)
        symbs.append(F)
        if G2 is not None: G_coupling.append(G2)

    G1 = sparse(G_converted)

    I,J,V = [],[],[]
    offset = [G_lq.size[0], 0]
    for Gcpl in G_coupling:
        I.append(Gcpl.I + offset[0])
        J.append(Gcpl.J + offset[1])
        V.append(Gcpl.V)
        offset[0] += Gcpl.size[0]
        offset[1] += Gcpl.size[1]
    G2 = spmatrix([v for v in itertools.chain(*V)],
                  [v for v in itertools.chain(*I)],
                  [v for v in itertools.chain(*J)],tuple(offset))
    
    if offset[0] == 0 or offset[1] == 0:
        G = G1
    else:
        G = sparse([[G1],[G2]])

    ct = matrix([c,matrix(0.0,(G2.size[1],1))])
    if A is not None:
        return (ct, G, matrix(h_converted),\
          {'l':dims['l'],'q':dims['q'],'s':dims_list},\
          sparse([[A],[spmatrix([],[],[],(A.size[0],G2.size[1]))]]),\
          b), block_to_sparse
    else:
        return (ct, G, matrix(h_converted),\
          {'l':dims['l'],'q':dims['q'],'s':dims_list}), block_to_sparse, symbs
	    
