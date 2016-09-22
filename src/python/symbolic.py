from cvxopt import matrix, spmatrix, normal, blas, printing
from chompack.misc import tril, perm, symmetrize
from chompack.misc import lmerge
from types import BuiltinFunctionType, FunctionType
            
def __tdfs(j, k, head, next, post, stack):
    """
    Depth-first search and postorder of a tree rooted at node j.
    """
    top = 0
    stack[0] = j
    while (top >= 0):
        p = stack[top]
        i = head[p]
        if i == -1:
            top -= 1
            post[k] = p
            k += 1
        else:
            head[p] = next[i]
            top += 1
            stack[top] = i

    return k

def post_order(parent):
    """
    Post order a forest.
    """
    n = len(parent)
    k = 0

    p = matrix(0,(n,1))
    head = matrix(-1,(n,1))
    next = matrix(0,(n,1))
    stack = matrix(0,(n,1))

    for j in range(n-1,-1,-1):
        if (parent[j] == j): continue
        next[j] = head[parent[j]]
        head[parent[j]] = j

    for j in range(n):
        if (parent[j] != j): continue
        k = __tdfs(j, k, head, next, p, stack)

    return p

def etree(A):
    """
    Compute elimination tree from upper triangle of A.
    """
    assert isinstance(A,spmatrix), "A must be a sparse matrix"
    assert A.size[0] == A.size[1], "A must be a square matrix"

    n = A.size[0]
    cp,ri,_ = A.CCS
    parent = matrix(0,(n,1))
    w = matrix(0,(n,1))

    for k in range(n):
        parent[k] = k
        w[k] = -1
        for p in range(cp[k],cp[k+1]):
            i = ri[p]
            while ((not i == -1) and (i < k)):
                inext = w[i]
                w[i] = k
                if inext == -1: parent[i] = k
                i = inext;

    return parent

def __leaf(i, j, first, maxfirst, prevleaf, ancestor):
    """
    Determine if j is leaf of i'th row subtree.
    """
    jleaf = 0
    if i<=j or first[j] <= maxfirst[i]: return -1, jleaf
    maxfirst[i] = first[j]
    jprev = prevleaf[i]
    prevleaf[i] = j
    if jprev == -1: jleaf = 1
    else: jleaf = 2
    if jleaf == 1: return i, jleaf
    q = jprev
    while q != ancestor[q]: q = ancestor[q]
    s = jprev
    while s != q:
        sparent = ancestor[s]
        ancestor[s] = q
        s = sparent
    return q, jleaf

def counts(A, parent, post):
    """
    Compute column counts.
    """
    n = A.size[0]
    colcount = matrix(0,(n,1))
    ancestor = matrix(range(n),(n,1))
    maxfirst = matrix(-1,(n,1))
    prevleaf = matrix(-1,(n,1))
    first = matrix(-1,(n,1))
    
    for k in range(n):
        j = post[k]
        if first[j] == -1:
            colcount[j] = 1
        else:
            colcount[j] = 0
        while  j != -1 and first[j] == -1:
            first[j] = k;
            j = parent[j]

    cp,ri,_ = A.CCS
    for k in range(n):
        j = post[k]
        if parent[j] != j:
            colcount[parent[j]] -= 1
        for p in range(cp[j],cp[j+1]):
            i = ri[p]
            if i <= j: continue
            q, jleaf = __leaf(i, j, first, maxfirst, prevleaf, ancestor)
            if jleaf >= 1: colcount[j] += 1
            if jleaf == 2: colcount[q] -= 1
        if parent[j] != j: ancestor[j] = parent[j]
    for j in range(n):
        if parent[j] != j: colcount[parent[j]] += colcount[j]

    return colcount

def pothen_sun(par, post, colcount):
    """
    Find supernodes and supernodal etree.

    ARGUMENTS
    par       parent array

    post      array with post ordering

    colcount  array with column counts

    RETURNS
    snpar     supernodal parent structure 

    flag      integer vector of length n; if flag[i] < 0, then -flag[i]
              is the degree of the supernode with repr. vertex i; if
              flag[i] >= 0, then flag[i] is the repr. vertex to which
              node i belongs.
    """
    
    n = len(par)
    flag = matrix(-1, (n, 1))
    snpar = matrix(-1, (n, 1))
    snodes = n
    ch = {}
    
    for j in post:

        if par[j] in ch: ch[par[j]].append(j)
        else: ch[par[j]] = [j]

        mdeg = colcount[j] - 1

        if par[j] != j:
            if mdeg == colcount[par[j]] and flag[par[j]] == -1:
                # par[j] not assigned to supernode
                snodes -= 1
                if flag[j] < 0:   # j is a repr. vertex
                    flag[par[j]] = j
                    flag[j] -= 1
                else:             # j is not a repr. vertex
                    flag[par[j]] = flag[j]
                    flag[flag[j]] -= 1
        else:
            if flag[j] < 0: snpar[j] = j
            else: snpar[flag[j]] = flag[j]

        if flag[j] < 0: k = j
        else: k = flag[j]

        if j in ch:
            for i in ch[j]:
                if flag[i] < 0: l = i
                else: l = flag[i]
                if not l == k: snpar[l] = k


    repr = matrix([i for i in range(n) if flag[i] < 0])
    deg = matrix([-flag[i] for i in range(n) if flag[i] < 0])

    # renumber etree with number of supernodes
    sn = matrix(-1, (n+1, 1))
    for k, r in enumerate(repr): sn[r] = k
    snpar = sn[snpar[repr]]

    return snpar, flag

def supernodes(par, post, colcount):
    """
    Find supernodes.

    ARGUMENTS
    par       parent array

    post      array with post ordering

    colcount  array with column counts

    RETURNS
    snode     array with supernodes; snode[snptr[k]:snptr[k+1]] contains
              the indices of supernode k

    snptr     pointer array; snptr[k] is the index of the representative
              vertex of supernode k in the snode array

    snpar     supernodal parent structure 
    """
    snpar, flag = pothen_sun(par, post, colcount)
    n = len(par)
    N = len(snpar)

    snode = matrix(0, (n,1))
    snptr = matrix(0, (N+1,1))

    slist = [[] for i in range(n)]
    for i in range(n):
        f = flag[i]
        if f < 0:
            slist[i].append(i)
        else:
            slist[f].append(i)

    k = 0; j = 0
    for i,sl in enumerate(slist):
        nsl = len(sl)
        if nsl > 0:
            snode[k:k+nsl] = matrix(sl)
            snptr[j+1] = snptr[j] + nsl
            k += nsl
            j += 1
        
    return snode, snptr, snpar

def amalgamate(colcount, snode, snptr, snpar, snpost, merge_function):
    """
    Supernodal amalgamation.

       colcount, snode, snptr, snpar, snpost = ...
         amalgamate(colcount, snode, snptr, snpar, snpost, merge_function)
    
    PURPOSE
    Iterates over the clique tree in topological order and greedily
    merges a supernode with its parent if

       merge_function(|J_{par(k)}|, |J_k|, |N_{par(k)}|, |N_k|)

    returns True.

    ARGUMENTS
    colcount  vector with column counts

    snode     vector with supernodes
 
    snptr     vector with offsets

    snpar     vector with supernodal parent indices

    snpost    vector with supernodal post ordering

    merge_function
              function 

    RETURNS
    colcount  vector with amalgamated column counts

    snode     vector with amalgamated supernodes
 
    snptr     vector with amalgamated offsets

    snpar     vector with amalgamated supernodal parent indices

    snpost    vector with amalgamated supernodal post ordering
    """
    N = len(snpost)
    ch = {}
    for j in snpost:
        if snpar[j] in ch: ch[snpar[j]].append(j)
        else: ch[snpar[j]] = [j]

    snlist = [snode[snptr[k]:snptr[k+1]] for k in range(N)]

    snpar_ = +snpar
    colcount_ = +colcount
    Ns = N
    for k in snpost:
        if snpar_[k] != k:
            colk = colcount_[snlist[k][0]]
            colp = colcount_[snlist[snpar_[k]][0]]
            nk = len(snlist[k])
            np = len(snlist[snpar_[k]])
            if merge_function and merge_function(colp,colk,np,nk):
                # merge supernode k and snpar[k]
                snlist[snpar_[k]] = matrix(sorted(list(snlist[k]) + list(snlist[snpar_[k]])))
                snlist[k] = None
                colcount_[snlist[snpar_[k]][0]] = colp + nk
                Ns -= 1
                if k in ch:
                    for c in ch[k]:
                        snpar_[c] = snpar_[k]
                    ch[snpar_[k]] += ch[k]
                snpar_[k] = k

    L = [i for i,s in enumerate(snlist) if s is not None]
    snptr_ = matrix(0,(len(L)+1,1))
    snode_ = +snode
    for i,l in enumerate(L):
        snptr_[i+1] = snptr_[i] + len(snlist[l])
        snode_[snptr_[i]:snptr_[i+1]] = snlist[l]

    snpar_ = snpar_[L]
    for i in range(len(snpar_)):
        snpar_[i] = L.index(snpar_[i])
    snpost_ = post_order(snpar_)
    return colcount_, snode_, snptr_, snpar_, snpost_

def embed(A, colcount, snode, snptr, snpar, snpost):
    """
    Compute filled pattern.

       colptr, rowidx = embed(A, colcount, snode, snptr, snpar, snpost)

    PURPOSE
    Computes rowindices and column pointer for representative vertices in supernodes.

    ARGUMENTS
    A         sparse matrix

    colcount  vector with column counts

    snode     vector with supernodes
 
    snptr     vector with offsets

    snpar     vector with supernodal parent indices

    snpost    vector with supernodal post ordering

    RETURNS
    colptr    vector with offsets 

    rowidx    vector with rowindices 
    """

    Alo = tril(A)
    cp,ri,_ = Alo.CCS
    N = len(snpar)

    # colptr for compressed cholesky factor
    colptr = matrix(0,(N+1,1))
    for k in range(N):
        colptr[k+1] = colptr[k] + colcount[snode[snptr[k]]]
    rowidx = matrix(-1,(colptr[-1],1))
    cnnz = matrix(0,(N,1))

    # compute compressed sparse representation
    for k in range(N):
        p = snptr[k]
        Nk = snptr[k+1]-p
        nk = cp[snode[p]+1] - cp[snode[p]]
        rowidx[colptr[k]:colptr[k]+nk] = ri[cp[snode[p]]:cp[snode[p]+1]]
        cnnz[k] = nk
        for i in range(1,Nk):
            nk = cp[snode[p+i]+1]-cp[snode[p+i]]
            cnnz[k] = lmerge(rowidx, ri, colptr[k], cp[snode[p+i]], cnnz[k], nk)

    for k in snpost:
        p = snptr[k]
        Nk = snptr[k+1]-p
        if snpar[k] != k:
            cnnz[snpar[k]] = lmerge(rowidx,rowidx,colptr[snpar[k]], colptr[k]+Nk,cnnz[snpar[k]], cnnz[k]-Nk)

    return colptr, rowidx

def relative_idx(colptr, rowidx, snptr, snpar):
    """
    Compute relative indices of update matrices in frontal matrix of parent.
    """
    
    relptr = matrix(0, (len(snptr),1))
    relidx = matrix(-1, (colptr[-1],1))

    def lfind(a,b):
        i = 0
        ret = +a
        for k in range(len(a)):
            while a[k] != b[i]: i += 1
            ret[k] = i
            i += 1
        return ret
    
    for k in range(len(snpar)):
        p = snpar[k]
        relptr[k+1] = relptr[k]
        if p != -1:
            nk = snptr[k+1] - snptr[k]
            relptr[k+1] += colptr[k+1] - colptr[k] - nk
            relidx[relptr[k]:relptr[k+1]] = lfind(rowidx[colptr[k]+nk:colptr[k+1]], rowidx[colptr[p]:colptr[p+1]])

    return relptr, relidx[:relptr[k+1]]

def peo(A, p):
    """
    Checks whether an ordering is a perfect elmimination order.

    Returns `True` if the permutation :math:`p` is a perfect elimination order
    for a Cholesky factorization :math:`PAP^T = LL^T`. Only the lower
    triangular part of :math:`A` is accessed.
    
    :param A:   :py:class:`spmatrix`
    
    :param p:   :py:class:`matrix` or :class:`list` of length `A.size[0]`
    """

    n = A.size[0]
    assert type(A) == spmatrix, "A must be a sparse matrix"
    assert A.size[1] == n, "A must be a square matrix"
    assert len(p) == n, "length of p must be equal to the order of A" 
    if isinstance(p, list): p = matrix(p)
    
    As = symmetrize(A)
    cp,ri,_ = As.CCS

    # compute inverse permutation array
    ip = matrix(0,(n,1))
    ip[p] = matrix(range(n),(n,1))

    # test set inclusion
    for k in range(n):
        v = p[k]  # next vertex to be eliminated

        # indices of neighbors that correspond to strictly lower triangular elements in reordered pattern
        r = set([rj for rj in ri[cp[v]:cp[v+1]] if ip[rj] > k])  

        for rj in r:
            if not r.issubset(set(ri[cp[rj]:cp[rj+1]])): return False
            
    return True

def merge_size_fill(tsize = 8, tfill = 8):
    """
    Simple heuristic for supernodal amalgamation (clique
    merging). 

    Returns a function that returns `True` if either (i) supernode k and
    supernode par(k) are both of size at most `tsize`, or (ii),
    merging supernodes par[k] and k induces at most `tfill` nonzeros
    in the lower triangular part of the sparsity pattern.

    :param tsize:   nonnegative integer; threshold for merging based on supernode sizes
    :param tfill:   nonnegative integer; threshold for merging based on induced fill
    """
    assert tsize >= 0, "tsize must be nonnegative"
    assert tfill >= 0, "tfill must be nonnegative"
    
    def fmerge(ccp,cck,np,nk):
        """
        Supernodal merge heuristic.

           d = fmerge(Jp, Jk, Np, Nk)

        PURPOSE
        Returns true if either (i) supernode k and supernode par(k)
        are both of size at most %i, or (ii), merging supernodes
        par(k) and k induces at most %i edges.

        ARGUMENTS
        Jp        integer; size of parent clique

        Jk        integer; size of child clique

        Np        integer; size of parent supernode 

        Nk        integer; size of child supernode
         
        RETURNS
        d         boolean
        """
        fill = (ccp - (cck - nk))*nk
        if fill <= tfill or (nk <= tsize and np <= tsize):
            return True
        else:
            return False
    # insert parameters into docstring of fmerge and return fmerge
    fmerge.__doc__ %= (tsize, tfill)
    return fmerge

class symbolic(object):
    """
    Symbolic factorization object.

    Computes symbolic factorization of a square sparse matrix
    :math:`A` and creates a symbolic factorization object.

    :param A:   :py:class:`spmatrix`
    :param p:   permutation vector or ordering routine (optional) 
    :param merge_function:  routine that implements a merge heuristic (optional)

    The optional argument `p` can be either a permutation vector or an
    ordering rutine that takes an :py:class:`spmatrix` and returns a
    permutation vector.

    The optional argument `merge_function` allows the user to
    merge supernodes in the elimination tree in a greedy manner;
    the argument must be a routine that takes the following four arguments and
    returns either `True` or `False`:
	
    :param cp:   clique order of the parent of clique :math:`k`
    :param ck:   clique order of clique :math:`k`
    :param np:   supernode order of the parent of supernode :math:`k`
    :param nk:   supernode order of supernode :math:`k`

    The clique k is merged with its parent if the return value is `True`.
    """
    
    def __init__(self, A, p = None, merge_function = None, **kwargs):

        assert isinstance(A,spmatrix), "A must be a sparse matrix"
        assert A.size[0] == A.size[1], "A must be a square matrix"

        supernodal = kwargs.get('supernodal',True)

        # Symmetrize A
        Ap = symmetrize(A)
        nnz_Ap = (len(Ap)+Ap.size[0])/2   # number of nonzeros in lower triangle        
        assert len([i for i,j in zip(Ap.I,Ap.J) if i==j]) == A.size[0], "the sparsity pattern of A must include diagonal elements" 

        # Permute if permutation vector p or ordering routine is specified
        if p is not None:
            if isinstance(p, BuiltinFunctionType) or isinstance(p, FunctionType):
                p = p(Ap)
            elif isinstance(p, list):
                p = matrix(p)
            assert len(p) == A.size[0], "length of permutation vector must be equal to the order of A"
            Ap = perm(Ap,p)

        # Symbolic factorization
        par = etree(Ap)
        post = post_order(par)
        colcount = counts(Ap, par, post)
        nnz_Ae = sum(colcount)
        if supernodal:
            snode, snptr, snpar = supernodes(par, post, colcount)
            snpost = post_order(snpar)
        else:
            snpar = par
            snpost = post
            snode = matrix(range(A.size[0]))
            snptr = matrix(range(A.size[0]+1))
        if merge_function:
            colcount, snode, snptr, snpar, snpost = amalgamate(colcount, snode, snptr, snpar, snpost, merge_function)

        # Post order nodes such that supernodes have consecutively numbered nodes
        pp = matrix([snode[snptr[snpost[k]]:snptr[snpost[k]+1]] for k in range(len(snpar))])
        snptr2 = matrix(0,(len(snptr),1))
        for k in range(len(snpar)):
            snptr2[k+1] = snptr2[k] + snptr[snpost[k]+1]-snptr[snpost[k]]
        colcount = colcount[pp]
        snposti = matrix(0,(len(snpost),1))
        snposti[snpost] = matrix(range(len(snpost)))
        snpar = matrix([snposti[snpar[snpost[k]]] for k in range(len(snpar)) ])        
        snode = matrix(range(len(snode)))
        snpost = matrix(range(len(snpost)))
        snptr = snptr2

        # Permute A and store effective permutation and its inverse
        Ap = perm(Ap,pp)
        if p is None:
            self.__p = pp
        else:
            self.__p = p[pp]
        self.__ip = matrix(0,(len(self.__p),1))
        self.__ip[self.__p] = matrix(range(len(self.__p)))

        # Compute embedding and relative indices
        sncolptr, snrowidx = embed(Ap, colcount, snode, snptr, snpar, snpost)
        relptr, relidx = relative_idx(sncolptr, snrowidx, snptr, snpar)
        
        # build chptr
        chptr = matrix(0, (len(snpar)+1,1))
        for j in snpost: 
            if snpar[j] != j: chptr[snpar[j]+1] += 1
        for j in range(1,len(chptr)):
            chptr[j] += chptr[j-1]

        # build chidx
        tmp = +chptr
        chidx = matrix(0,(chptr[-1],1))
        for j in snpost:
            if snpar[j] != j: 
                chidx[tmp[snpar[j]]] = j
                tmp[snpar[j]] += 1
        del tmp

        # compute stack size
        stack_size = 0
        stack_mem = 0
        stack_max = 0
        frontal_max = 0
        stack = []
        for k in snpost:
            nn = snptr[k+1]-snptr[k]
            na = relptr[k+1]-relptr[k]
            nj = na + nn
            frontal_max = max(frontal_max, nj**2)
            for j in range(chptr[k+1]-1,chptr[k]-1,-1):
                v = stack.pop()
                stack_mem -= v**2
            if (na > 0):
                stack.append(na)
                stack_mem += na**2
                stack_max = max(stack_max,stack_mem)
                stack_size = max(stack_size,len(stack))
        self.frontal_len = frontal_max
        self.stack_len = stack_max
        self.stack_size = stack_size
                
        # build blkptr
        blkptr = matrix(0, (len(snpar)+1,1))
        for i in range(len(snpar)):
            blkptr[i+1] = blkptr[i] + (snptr[i+1]-snptr[i])*(sncolptr[i+1]-sncolptr[i])

        # compute storage requirements
        stack = []
        stack_depth = 0

        stack_mem = 0
        stack_tmp = 0
        cln = 0

        stack_solve = 0
        stack_stmp = 0

        for k in snpost:
            nn = snptr[k+1]-snptr[k]       # |Nk|
            na = relptr[k+1]-relptr[k]     # |Ak|
            nj = na + nn
            cln = max(cln,nj)              # this is the clique number
            for i in range(chptr[k+1]-1,chptr[k]-1,-1):
                na_ch = stack.pop()
                stack_tmp -= na_ch**2
                stack_stmp -= na_ch
            if na > 0:
                stack.append(na)
                stack_tmp += na**2
                stack_mem = max(stack_tmp,stack_mem)
                stack_stmp += na
                stack_solve = max(stack_stmp,stack_solve)
            stack_depth = max(stack_depth,len(stack))
            
        self.__clique_number = cln
        self.__n = len(snode)
        self.__Nsn = len(snpar)
        self.__snode = snode
        self.__snptr = snptr
        self.__chptr = chptr
        self.__chidx = chidx
        self.__snpar = snpar
        self.__snpost = snpost
        self.__relptr = relptr
        self.__relidx = relidx
        self.__sncolptr = sncolptr
        self.__snrowidx = snrowidx
        self.__blkptr = blkptr
        self.__fill = (nnz_Ae-nnz_Ap,self.nnz-nnz_Ae)
        self.__memory = {'stack_depth':stack_depth,
                         'stack_mem':stack_mem,
                         'frontal_mem':cln**2,                         
                         'stack_solve':stack_solve}

        return

    def __repr__(self):
        return "<symbolic factorization, n=%i, nnz=%i, nsn=%i>" % (self.n,self.nnz,self.Nsn)

    def __str__(self):
        opts = printing.options
        printing.options = {'iformat':'%1i','dformat':'%1.0f',\
                            'width':printing.options['width'],'height':printing.options['height']}
        tmp = self.sparsity_pattern(reordered = True, symmetric = True).__str__()
        printing.options = opts
        return tmp.replace('0',' ').replace('1','X')

    def sparsity_pattern(self, reordered = True, symmetric = True):
        """
        Returns a sparse matrix with the filled pattern. By default,
        the routine uses the reordered pattern, and the inverse
        permutation is applied if `reordered` is `False`.
        
        :param reordered:  boolean (default: `True`)
        :param symmetric:  boolean (default: `True`)	
        """
        return cspmatrix(self, 1.0).spmatrix(reordered = reordered, symmetric = symmetric)

    @property
    def n(self):
        """Number of nodes (matrix order)"""
        return self.__n

    @property
    def Nsn(self):
        """Number of supernodes"""
        return self.__Nsn

    @property
    def snode(self):
        """
        Supernode array: supernode :math:`k` consists of nodes
        `snode[snptr[k]:snptr[k+1]]` where `snptr` is the supernode
        pointer array
        """
        return self.__snode

    @property
    def snptr(self):
        """
        Supernode pointer array: supernode :math:`k` is of order
        `snpptr[k+1]-snptr[k]` and supernode :math:`k` consists of nodes
        `snode[snptr[k]:snptr[k+1]]`
        """
        return self.__snptr

    @property
    def chptr(self):
        """
        Pointer array associated with `chidx`:
        `chidx[chptr[k]:chptr[k+1]]` are the indices of the children
        of supernode k.
        """
        return self.__chptr

    @property
    def chidx(self):
        """
        Integer array with indices of child vertices in etree: 
        `chidx[chptr[k]:chptr[k+1]]` are the indices of the children
        of supernode :math:`k`.
        """
        return self.__chidx


    @property
    def snpar(self):
        """
        Supernode parent array: supernode :math:`k` is a root of the
        supernodal elimination tree if `snpar[k]` is equal to k, and
        otherwise `snpar[k]` is the index of the parent of supernode
        :math:`k` in the supernodal elimination tree
        """
        return self.__snpar

    @property
    def snpost(self):
        """Supernode post-ordering"""
        return self.__snpost

    @property
    def relptr(self):
        """Pointer array assoicated with `relidx`."""
        return self.__relptr

    @property 
    def relidx(self): 
        """ The relative index array facilitates fast "extend-add" and
        "extract" operations in the supernodal-multifrontal
        algorithms. The relative indices associated with supernode
        :math:`k` is a list of indices :math:`I` such that the frontal
        matrix :math:`F` associated with the parent of node :math:`k`
        can be updated as `F[I,I] += Uj`. The relative indices are
        stored in an integer array `relidx` with an associated pointer
        array `relptr`."""
        return self.__relidx

    @property
    def sncolptr(self):
        """
        Pointer array associated with `snrowidx`.
        """
        return self.__sncolptr

    @property
    def snrowidx(self):
        """
        Row indices associated with representative vertices:
        `snrowidx[sncolptr[k]:sncolptr[k+1]]` are the row indices in
        the column corresponding the the representative vertex of
        supernode :math:`k`, or equivalently,
        `snrowidx[sncolptr[k]:sncolptr[k+1]]` is the :math:`k`'th
        clique.
        """
        return self.__snrowidx

    @property
    def blkptr(self):
        """
        Pointer array for block storage of chordal sparse matrix.
        """        
        return self.__blkptr

    @property
    def fill(self):
        """
        Tuple with number of lower-triangular fill edges: `fill[0]` is
        the fill due to symbolic factorization, and `fill[1]` is the
        fill due to supernodal amalgamation"""
        return self.__fill
    
    @property
    def nnz(self):
        """
        Returns the number of lower-triangular nonzeros.
        """
        nnz = 0
        for k in range(len(self.snpost)):
            nn = self.snptr[k+1]-self.snptr[k]    
            na = self.relptr[k+1]-self.relptr[k]
            nnz += nn*(nn+1)/2 + nn*na
        return nnz

    @property
    def p(self):
        """
        Permutation vector
        """
        return self.__p

    @property
    def memory(self):
        return self.__memory

    @property
    def ip(self):
        """
        Inverse permutation vector
        """
        return self.__ip

    @property
    def clique_number(self):
        """
        The clique number (the order of the largest clique)
        """
        return self.__clique_number

    
    def cliques(self, reordered = True):
        """
        Returns a list of cliques
        """
        if reordered:
            return [list(self.snrowidx[self.sncolptr[k]:self.sncolptr[k+1]]) for k in range(self.Nsn)]
        else:
            return [list(self.__p[self.snrowidx[self.sncolptr[k]:self.sncolptr[k+1]]]) for k in range(self.Nsn)]

    def separators(self, reordered = True):
        """
        Returns a list of separator sets 
        """
        if reordered:
            return [list(self.snrowidx[self.sncolptr[k]+self.snptr[k+1]-self.snptr[k]:self.sncolptr[k+1]]) for k in range(self.Nsn)] 
        else:
            return [list(self.__p[self.snrowidx[self.sncolptr[k]+self.snptr[k+1]-self.snptr[k]:self.sncolptr[k+1]]]) for k in range(self.Nsn)]
        
    def supernodes(self, reordered = True):
        """
        Returns a list of supernode sets
        """
        if reordered:
            return [list(self.snode[self.snptr[k]:self.snptr[k+1]]) for k in range(self.Nsn)]
        else:
            return [list(self.__p[self.snode[self.snptr[k]:self.snptr[k+1]]]) for k in range(self.Nsn)]
        
    def parent(self):
        """
        Returns a supernodal parent list: the i'th element is equal to -1 if 
        supernode i is a root node in the clique forest, and otherwise
        the i'th element is the index of the parent of supernode i.
        """
        return list(self.snpar)
        
class cspmatrix(object):
    """
    Chordal sparse matrix object.

    :param symb:    :py:class:`symbolic` object
    :param blkval:  :py:class:`matrix` with numerical values (optional)
    :param factor:  boolean (default is `False`)
    
    A  :py:class:`cspmatrix` object contains a reference to a symbolic
    factorization as well as an array with numerical values which are
    stored in a compressed block storage format which is a block variant
    of the compressed column storage format.

    `A = cspmatrix(symb)` creates a new chordal sparse matrix object
    with a sparsity pattern defined by the symbolic factorization
    object `symb`. If the optional argument `blkval` specified, the
    :py:class:`cspmatrix` object will use the `blkval` array for numerical values
    (and not a copy!), and otherwise the  :py:class:`cspmatrix` object is
    initialized with an all-zero array. The optional input `factor`
    determines whether or not the :py:class:`cspmatrix` stores a factored
    matrix. 
    """

    def __init__(self, symb, blkval = None, factor = False):

        assert isinstance(symb, symbolic), "symb must be an instance of symbolic"
        self._is_factor = factor   

        self.symb = symb          # keep a reference to symbolic object
        if blkval is None:
            # initialize cspmatrix with zeros
            self.blkval = matrix(0.0, (symb.blkptr[-1],1))
        elif type(blkval) is float:
            self.blkval = matrix(blkval, (symb.blkptr[-1],1))
        else:
            assert len(blkval) == symb.blkptr[-1], "dimension mismatch: length of blkval is incorrect"
            self.blkval = blkval

    def __str__(self):
        return self.spmatrix(reordered = True, symmetric = False).__str__()

    def __repr__(self):
        nnz = 0
        for k in range(self.symb.Nsn):
            nn = self.symb.snptr[k+1]-self.symb.snptr[k]  
            na = self.symb.relptr[k+1]-self.symb.relptr[k]
            nnz += nn*na + nn*(nn+1)/2
        if self.is_factor:
            return "<%ix%i chordal sparse matrix (factor), tc='%s', nnz=%i, nsn=%i>"\
              % (self.symb.n,self.symb.n,self.blkval.typecode,nnz,self.symb.Nsn) 
        else:
            return "<%ix%i chorcal sparse matrix, tc='%s', nnz=%i, nsn=%i>"\
              % (self.symb.n,self.symb.n,self.blkval.typecode,nnz,self.symb.Nsn)
        
    def __iadd__(self, B):
        assert self.is_factor is False, "Addition of cspmatrix factors not supported"
        if isinstance(B, cspmatrix):
            assert self.symb == B.symb, "Symbolic factorization mismatch"
            assert B.is_factor is False, "Addition of cspmatrix factors not supported"
            blas.axpy(B.blkval, self.blkval)
        elif isinstance(B, spmatrix):
            self._iadd_spmatrix(B)
        else:
            raise TypeError 
        return self

    def __isub__(self, B):
        assert self.is_factor is False, "Addition of cspmatrix factors not supported"
        if isinstance(B, cspmatrix):
            assert self.symb == B.symb, "Symbolic factorization mismatch"
            assert B.is_factor is False, "Addition of cspmatrix factors not supported"
            blas.axpy(B.blkval, self.blkval, alpha = -1.0)
        elif isinstance(B, spmatrix):
            self._iadd_spmatrix(B, alpha = -1.0)    
        else:
            raise TypeError 
        return self

    def __add__(self, B):
        assert self.is_factor is False, "Addition of cspmatrix factors not supported"
        if isinstance(B, cspmatrix):
            assert self.symb == B.symb, "Symbolic factorization mismatch"
            assert B.is_factor is False, "Addition of cspmatrix factors not supported"
            C = self.copy()
            C.__iadd__(B)
        elif  isinstance(B, spmatrix):
            assert self.is_factor is False
            C = self.copy()
            C._iadd_spmatrix(B)
        else:
            raise TypeError
        return C

    def __sub__(self, B):
        assert self.is_factor is False, "Addition of cspmatrix factors not supported"
        if isinstance(B, cspmatrix):
            assert self.symb == B.symb, "Symbolic factorization mismatch"
            assert B.is_factor is False, "Addition of cspmatrix factors not supported"
            C = self.copy()
            C.__isub__(B)
        elif  isinstance(B, spmatrix):
            C = self.copy()
            C._iadd_spmatrix(B, alpha = -1.0)    
        else:
            raise TypeError
        return C

    def __imul__(self, a):
        if isinstance(a,float):
            blas.scal(a, self.blkval)
        elif isinstance(a, int) or isinstance(a, long):
            blas.scal(float(a), self.blkval)
        else:
            raise NotImplementedError("only scalar multiplication has been implemented")
        return self
    
    def __mul__(self, a):
        ret = self.copy()
        ret.__imul__(a)
        return ret

    def __rmul__(self, a):
        return self.__mul__(a)
                    
    @property
    def is_factor(self):
        """
        This property is equal to `True` if the cspmatrix represents a
        Cholesky factor, and otherwise it is equal to `False`.
        """
        return self._is_factor
    
    @is_factor.setter
    def is_factor(self, value):
        self._is_factor = value

    @property
    def size(self):
        return (self.symb.n,self.symb.n)
            
    def spmatrix(self, reordered = True, symmetric = False):
        """
        Converts the :py:class:`cspmatrix` :math:`A` to a sparse matrix. A reordered
        matrix is returned if the optional argument `reordered` is
        `True` (default), and otherwise the inverse permutation is applied. Only the
        default options are allowed if the :py:class:`cspmatrix` :math:`A` represents
        a Cholesky factor. 

        :param reordered:  boolean (default: True)
        :param symmetric:  boolean (default: False)			   
        """
        n = self.symb.n
        snptr = self.symb.snptr
        snode = self.symb.snode
        relptr = self.symb.relptr
        snrowidx = self.symb.snrowidx
        sncolptr = self.symb.sncolptr
        blkptr = self.symb.blkptr
        blkval = self.blkval
        
        if self.is_factor:
            if symmetric: raise ValueError("'symmetric = True' not implemented for Cholesky factors")
            if not reordered: raise ValueError("'reordered = False' not implemented for Cholesky factors")
            snpost = self.symb.snpost
            blkval = +blkval
            for k in snpost:
                j = snode[snptr[k]]            # representative vertex
                nn = snptr[k+1]-snptr[k]       # |Nk|
                na = relptr[k+1]-relptr[k]     # |Ak|
                if na == 0: continue
                nj = na + nn
                if nn == 1:
                    blas.scal(blkval[blkptr[k]],blkval,offset = blkptr[k]+1,n=na)
                else:
                    blas.trmm(blkval,blkval, transA = "N", diag = "N", side = "R",uplo = "L", \
                              m = na, n = nn, ldA = nj, ldB = nj, \
                              offsetA = blkptr[k],offsetB = blkptr[k] + nn)

        cc = matrix(0,(n,1))  # count number of nonzeros in each col
        for k in range(self.symb.Nsn):
            nn = snptr[k+1]-snptr[k]
            na = relptr[k+1]-relptr[k]
            nj = nn + na
            for i in range(nn):
                j = snode[snptr[k]+i]
                cc[j] = nj - i

        # build col. ptr
        cp = [0]
        for i in range(n): cp.append(cp[-1] + cc[i])
        cp = matrix(cp)

        # copy data and row indices
        val = matrix(0.0, (cp[-1],1))
        ri = matrix(0, (cp[-1],1))
        for k in range(self.symb.Nsn):
            nn = snptr[k+1]-snptr[k]
            na = relptr[k+1]-relptr[k]
            nj = nn + na
            for i in range(nn):
                j = snode[snptr[k]+i]
                blas.copy(blkval, val, offsetx = blkptr[k]+nj*i+i, offsety = cp[j], n = nj-i)
                ri[cp[j]:cp[j+1]] = snrowidx[sncolptr[k]+i:sncolptr[k+1]]

        I = []; J = []
        for i in range(n):
            I += list(ri[cp[i]:cp[i+1]])
            J += (cp[i+1]-cp[i])*[i]

        tmp = spmatrix(val, I, J, (n,n))  # tmp is reordered and lower tril.
        
        if reordered or self.symb.p is None:
            # reordered matrix (do not apply inverse permutation)
            if not symmetric: return tmp
            else: return symmetrize(tmp)
        else:
            # apply inverse permutation            
            tmp = perm(symmetrize(tmp), self.symb.ip)
            if symmetric: return tmp
            else: return tril(tmp) 

    def diag(self, reordered = True):
        """
        Returns a vector with the diagonal elements of the matrix.
        """
        sncolptr = self.symb.sncolptr
        snptr = self.symb.snptr
        snode = self.symb.snode
        blkptr = self.symb.blkptr
        
        D = matrix(0.0,(self.symb.n,1))
        for k in range(self.symb.Nsn):
            nn = snptr[k+1]-snptr[k]
            w = sncolptr[k+1]-sncolptr[k]
            for i in range(nn): D[snode[snptr[k]+i]] = self.blkval[blkptr[k]+i*w+i]
        if reordered: return D
        else: return D[self.symb.ip] 
                
    def copy(self):
        """
        Returns a new :py:class:`cspmatrix` object with a reference to the same
        symbolic factorization, but with a copy of the array
        that stores the numerical values.
        """
        return cspmatrix(self.symb, blkval = +self.blkval, factor = self.is_factor)
    
    def _iadd_spmatrix(self, X, alpha = 1.0):
        """
        Add a sparse matrix :math:`X` to :py:class:`cspmatrix`.
        """
        assert self.is_factor is False, "cannot add spmatrix to a cspmatrix factor"

        n = self.symb.n
        snptr = self.symb.snptr
        snode = self.symb.snode
        relptr = self.symb.relptr
        snrowidx = self.symb.snrowidx
        sncolptr = self.symb.sncolptr
        blkptr = self.symb.blkptr
        blkval = self.blkval

        if self.symb.p is not None:
            Xp = tril(perm(symmetrize(X),self.symb.p))
        else:
            Xp = tril(X)
        cp, ri, val = Xp.CCS

        # for each block ...
        for k in range(self.symb.Nsn):
            nn = snptr[k+1]-snptr[k]
            na = relptr[k+1]-relptr[k]
            nj = nn + na

            r = list(snrowidx[sncolptr[k]:sncolptr[k+1]])
            # copy cols from A to block
            for i in range(nn):
                j = snode[snptr[k]+i]
                offset = blkptr[k] + nj*i
                # extract correct indices and add values
                I = [offset + r.index(idx) for idx in ri[cp[j]:cp[j+1]]]

                blkval[I] += alpha*val[cp[j]:cp[j+1]]
        return

    def add_projection(self, A, alpha = 1.0, beta = 1.0, reordered=False):
        """
        Add projection of a dense matrix :math:`A` to :py:class:`cspmatrix`.

            X := alpha*proj(A) + beta*X
        """
        assert self.is_factor is False, "cannot project matrix onto a cspmatrix factor"
        assert isinstance(A, matrix), "argument A must be a dense matrix"
        
        symb = self.symb
        blkval = self.blkval

        n = symb.n
        snptr = symb.snptr
        snode = symb.snode
        relptr = symb.relptr
        snrowidx = symb.snrowidx
        sncolptr = symb.sncolptr
        blkptr = symb.blkptr

        if self.symb.p is not None and reordered is False:
            A = tril(A)
            A = A+A.T
            A[::A.size[0]+1] *= 0.5
            A = A[self.symb.p,self.symb.p]

        # for each block ...
        for k in range(self.symb.Nsn):
            nn = snptr[k+1]-snptr[k]
            na = relptr[k+1]-relptr[k]
            nj = nn + na
             
            blkval[blkptr[k]:blkptr[k+1]] = beta*blkval[blkptr[k]:blkptr[k+1]] + alpha*(A[snrowidx[sncolptr[k]:sncolptr[k+1]],snode[snptr[k]:snptr[k+1]]][:])
        
        return 
