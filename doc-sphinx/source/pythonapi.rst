.. _c-pythonapi:

****************
Python Interface
****************

The Python interface extends the Python 
`CVXOPT <http://abel.ee.ucla.edu/cvxopt>`_
package with functions from the CHOMPACK library.
It is based on the CVXOPT dense and sparse matrix types,
and two new Python objects: a *chordal matrix* object for storing 
symmetric chordal matrices, and a *chordal factor* object for storing 
Cholesky factors of positive definite chordal matrices.  These two data 
types can be created and manipulated, and converted to CVXOPT matrices.
using the functions described below.  They can be thought of as opaque 
objects that contain the values of the matrix, plus the sparsity pattern
and a perfect elimination ordering for the sparsity pattern.

When the following documentation states a requirement that two chordal 
matrix objects and/orfactors have the same sparsity pattern, we mean by 
this that they were created from the same chordal matrix object via a 
series of calls to functions that create a new chordal matrix or factor 
from an existing one (such as :func:`cholesky`, :func:`completion`, 
:func:`copy`, :func:`llt`, :func:`partial_inv`, or :func:`project`).  
Chordal matrices that were created from CVXOPT matrices via the function 
:func:`embed` are not recognized to have the same sparsity pattern, even 
though their sparsity patterns may be equal (mathematically).

CVXOPT integer matrices are used to represent permutation matrices.  The 
relation between the CVXOPT integer matrix ``p`` (a permutation of the 
column matrix with entries `0`, `1`, ..., `n-1`) and the permutation matrix
:math:`P` it represents is as follows: if the CVXOPT matrix ``X`` has 
value :math:`X`, then the CVXOPT matrix ``X[p, p]`` has value :math:`P^TXP`.


Functions
=========

The functions in the package can be divided in different groups.

1. Conversion from CVXOPT matrices to chordal matrix objects and vice-versa.

   * :func:`embed`: finds a chordal embedding of the sparsity pattern of a
     non-chordal symmetric sparse CVXOPT matrix, projects the matrix
     on the embedding, and returns the result as a chordal matrix.
   * :func:`project`: projects a non-chordal symmetric sparse CVXOPT matrix
     on a given chordal sparsity pattern, and returns the result as a
     chordal matrix.
   * :func:`sparse`: converts a chordal matrix or factor to a CVXOPT 
     sparse matrix.

2. Main computational routines.

   * :func:`cholesky`: Cholesky factorization of a positive definite 
     symmetric chordal matrix.
   * :func:`solve`: multiplication with a Cholesky factor or its inverse.
   * :func:`completion`: maximum determinant positive definite completion of
     of a symmetric chordal matrix.
   * :func:`partial_inv`: projection of the inverse of a positive definite
     chordal matrix on its sparsity pattern.
   * :func:`hessian`: evaluates the hessian or inverse hessian of the
     logarithmic barrier function for the positive definite matrices with 
     a given chordal sparsity pattern.

3. Auxiliary routines for chordal matrices.

   * :func:`copy`: makes a copy of a chordal matrix.
   * :func:`scal`: scales a chordal matrix by a scalar.
   * :func:`axpy` adds a multiple of a chordal matrix to a chordal 
     matrix with the same sparsity pattern.
   * :func:`dot`: inner product of two chordal matrices with the same
     sparsity pattern.
   * :func:`syr2`: computes projected rank 2 update of a chordal matrix.
   * :func:`llt`: computes a chordal matrix from its Cholesky factor.
   * :func:`logdet`: returns log(det(L)) for a Cholesky factor L.
   * :func:`info`: returns a dictionary with information about a chordal 
     sparsity pattern.

4. Auxiliary routines for CVXOPT sparse matrices.

   * :func:`symmetrize`: computes X + X' - diag(diag(X)) for a lower
     triangular sparse CVXOPT matrix X.
   * :func:`perm`: an efficient method for computing a symmetric reordering
     X[p, p] of a CVXOPT sparse matrix X.
   * :func:`tril`: returns the lower triangular part of a square CVXOPT
     sparse matrix.
   * :func:`peo`: checks whether a given permutation is a perfect 
     elimination ordering for a CVXOPT sparse matrix.
   * :func:`maxcardsearch`: returns the maximum cardinality search 
     reordering of a chordal sparse CVXOPT matrix.
   * :func:`maxchord`: computes a maximal chordal subgraph from a 
     sparse  CVXOPT matrix and returns chordal matrix.

Conversion to/from CVXOPT
=========================

.. function:: embed(X[, p = None])

    Computes a chordal embedding of a sparse matrix.

    :samp:`Y, nfill = embed(X, p = None)`

    Returns a chordal embedding of the sparsity pattern of ``X``, projects
    ``X`` on the embedding, and returns the result as a chordal matrix
    object.  The argument ``p`` is a permutation with as default value 
    the natural ordering :samp:`matrix([0, 1, ..., n-1])`.  The embedding 
    is computed via a symbolic Cholesky factorization of :samp:`X[p, p]`.
    
    :param X: CVXOPT sparse square matrix of doubles.  Only the lower
              triangular part of the matrix is accessed
    :param p: CVXOPT dense integer matrix of length `n`, if `n` is the order
              of *X*
    :returns: tuple (*Y*, *nfill*) with ``Y`` a chordal matrix embedding  
              and ``nfill`` the number of nonzero entries added in the 
              embedding


.. function:: project(X, Y)

    Projects a CVXOPT sparse matrix on a chordal sparsity pattern.

    :samp:`C = project(X, Y)`

    Projects the CVXOPT sparse matrix ``Y`` on the sparsity pattern of the 
    chordal matrix ``X``, and returns the result as a chordal matrix.  
    Only the lower triangular part of ``X`` is referenced.

    :param X: chordal matrix
    :param Y: square CVXOPT sparse matrix of doubles
    :return: chordal matrix with the same sparsity pattern as ``X``


.. function:: sparse(X)

    Converts a chordal matrix or factor to a CVXOPT sparse matrix.

    :samp:`L = sparse(X)`
    
    If ``X`` is a chordal matrix, the function returns the lower triangular 
    part of ``X[p, p]`` as a CVXOPT sparse matrix.  If ``X`` is 
    a chordal factor for a Cholesky factorization :eq:`e-chol` the 
    function returns the lower triangular sparse matrix ``L``.

    :param X: chordal matrix or factor
    :returns: lower triangular CVXOPT sparse square matrix of doubles


Computational routines
======================

.. function:: cholesky(X)

    Cholesky factorization.
    
    :samp:`L = cholesky(X)`
    
    Computes a zero fill-in Cholesky factorization 

    .. math:: 
        :label: e-chol
        
        P^TXP = LL^T

    of a positive definite chordal matrix ``X``. 

    :param X: chordal matrix
    :returns: chordal factor if the factorization was successful
    :raises: an `ArithmethicError` if the matrix  is not positive definite


.. function:: solve(L, X[, mode = 0])

    Solves a factored set of equations, or multiplies with Cholesky factors

    :samp:`solve(L, X, mode = 0)`

    ``L`` contains the factors of a factorization :eq:`e-chol` of a 
    positive definite sparse chordal matrix.  ``X`` is a CVXOPT dense matrix
    of doubles with the same number of rows as ``L``. 
    On exit, ``X`` is overwritten with one of the four matrices in the 
    table.

    .. tabularcolumns:: |l|c|
    
    =======================   ====
    action                    mode
    =======================   ====
    :math:`X := L^{-1}P^TX`   0
    :math:`X := PL^{-T}X`     1
    :math:`X := PLX`          2
    :math:`X := L^TP^TX`      3
    =======================   ====

    :param L: chordal factor
    :param X: CVXOPT dense matrix of doubles with `n` rows if `n` is the
              order of `L`
    :param mode: integer

    
.. function:: completion(X)

    Maximum-determinant positive definite completion. 
    
    :samp:`L = completion(X)`
    
    Returns the Cholesky factor of the inverse of the maximum-determinant 
    positive definite completion of a symmetric chordal matrix ``X``, \ie,
    the Cholesky factor of the inverse of the solution of

    .. math::

        \begin{array}{ll}
        \mbox{maximize}   &  \det W \\
        \mbox{subject to} &  \mbox{proj}(W) = X \\
                          &  W \succ 0.
        \end{array}

    The inverse :math:`Z = W^{-1}` has the same sparsity pattern as 
    :math:`X` and satisfies the nonlinear equation 

    .. math::

        \mbox{proj}(Z^{-1}) = X. 

    :func:`completion` returns the factor in the factorization 
    :math:`P^T Z P = L L^T`.
    
    :param X: chordal matrix
    :returns: chordal factor with the same sparsity pattern as `X`
    :raises:  an `ArithmethicError` if the matrix does not have a positive 
              definite completion


.. function:: partial_inv(L)

    Evaluates the projection of the inverse of the matrix 
    :math:`X = PLL^TP^T` on the sparsity pattern of :math:`X`.

    :samp:`Y = partial_inv(L)`

    Computes 

    .. math::

        Y = \mathrm{proj}(X^{-1}) 

    where :math:`X` is a positive definite chordal matrix specified by
    its Cholesky factor :math:`P^TXP = LL^T`.

    :param L: chordal factor
    :returns: chordal matrix with the same sparsity pattern as ``L``


.. function:: hessian(L, Y, U[, adj = False[, inv = False]])

    The mapping 
    
    .. math::
    
       \mathcal{H}_X(U) = \mathrm{proj}(X^{-1} U X^{-1})

    is the Hessian of the log-det barrier at a positive definite chordal
    matrix :math:`X`, applied to a symmetric chordal matrix :math:`U`.
    The Hessian operator can be factored as

    .. math::
    
        \mathcal H_X(U) = \mathcal G_X^\mathrm{adj} ( \mathcal G_X( U )),

    where the mappings on the right hand side are adjoint mappings that map
    chordal symmetric matrices to chordal symmetric matrices. 

    :samp:`hessian(L, Y, U, adj = False, inv = False)`

    evaluates these mappings or their inverses for a list of symmetric 
    chordal matrices ``U = [ U[0], ..., U[N-1] ]``, and overwrites the
    matrices with the results.   The following table lists the possible 
    actions.
    
    .. tabularcolumns:: |L|L|L|   FIXME: This has no effect, opposed to Sphinx doc.
    
    ================================================  =====  =====
    Action                                            adj    inv
    ================================================  =====  =====
    :math:`U_i:={\cal G}_X(U_i)`                      False  False
    :math:`U_i:={\cal G}_X^{-1}(U_i)`                 False  True
    :math:`U_i:={\cal G}_X^\mathrm{adj}(U_i)`         True   False
    :math:`U_i:=({\cal G}_X^\mathrm{adj})^{-1}(U_i)`  True   True  
    ================================================  =====  =====
    
    The input argument ``L`` is the Cholesky factor of :math:`X`,  
    as computed by the command :samp:`L = cholesky(X)`.
    The input argument ``Y`` is the partial inverse of the inverse of
    :math:`X`, as computed by the command :samp:`Y = partial_inv(L)`.
    The input argument ``U`` is a list of chordal matrices with the 
    same sparsity pattern as ``L`` and ``Y``.

    The matrices :math:`\mathcal H_X(U_i)` can be computed by two calls

    ::

        hessian(L, Y, U, adj = False, inv = False)
        hessian(L, Y, U, adj = True, inv = False)

    The matrices :math:`\mathcal H_X^{-1}(U_i)` can be computed as

    ::

        hessian(L, Y, U, adj = True, inv = True)
        hessian(L, Y, U, adj = False, inv = True)

    :param L: chordal factor
    :param Y: chordal matrix with the same sparsity pattern as ``L``.
    :param U: list of chordal matrices with the same sparsity pattern
        as ``L`` and ``Y``
    :param adj: True/False
    :param inv: True/False




Auxiliary routines for chordal matrices
=======================================

.. function:: copy(X)

    Returns a copy of a chordal matrix.
    
    :param X: chordal matrix
    :returns: chordal matrix with the same sparsity pattern and 
              numerical values as ``X``


.. function:: scal(alpha, X)

    Evaluates 

    .. math:: 
    
        X := \alpha X.
 
    :param alpha: scaling factor
    :param X: chordal matrix


.. function:: axpy(X, Y, alpha)
    
    Evaluates 

    .. math::

        Y := \alpha X + Y.
    
    :param X: chordal matrix
    :param Y: chordal matrix with the same sparsity pattern as ``X``
    :param alpha: float


.. function:: dot(X, Y)

    Inner product of symmetric chordal sparse matrices.

    Returns the inner product 

    .. math::

        \mathrm{tr}(XY)

    of two symmetric sparse matrices with the same chordal sparsity pattern.
    
    :param X: chordal matrix
    :param Y: chordal matrix with have the same sparsity pattern as ``X``
    :rtype: float
    

.. function:: syr2(X, y, z, alpha=1.0, beta=1.0)

    Computes projected rank 2 update of a chordal matrix 

    .. math::

        X := \alpha \mathrm{proj}(yz^T + zy^T) + \beta X

    where :math:`X` is of order `n`, and :math:`y` and :math:`z` are `n`-vectors.
    
    :param X: chordal matrix
    :param y: CVXOPT dense matrix of doubles of length ``n``
    :param z: CVXOPT dense matrix of doubles of length ``n``
    :param alpha: float
    :param beta: float
    :rtype: float


.. function:: llt(L)

    Computes a symmetric matrix from its Cholesky factorization
    
    :samp:`X = llt(L)`
    
    Computes X from its Cholesky factorization :math:`P^T X P = L L^T`.
    
    :param L: chordal factor
    :returns: chordal matrix with the same sparsity pattern as ``L``

.. function:: logdet(L)

    Returns the logarithm of the determinant of a Cholesky factor ``L``.

    :param L: chordal factor
    :returns: float
    

.. function:: info(X)

    Returns a dictionary with information about a chordal sparsity pattern.
    
    :param X: chordal matrix or factor
    :returns: python dictionary representation of the sparsity pattern



Auxiliary routines for CVXOPT matrices
======================================

    
    
.. function:: maxcardsearch(X)

    Maximum cardinality search ordering of a sparse chordal matrix.

    Returns the maximum cardinality search ordering of a symmetric chordal 
    matrix ``X``.  The maximum cardinality search ordering is a perfect 
    elimination ordering for the Cholesky factorization.

    :param X: CVXOPT sparse square matrix of doubles.  Only the sparsity
        pattern of the lower triangular part of the matrix is accessed

    :returns: CVXOPT dense integer matrix of length ``n``, if ``n`` is the 
        order of ``X``


.. function:: peo(X, p)

    Checks whether an ordering is a perfect elmimination order.

    Returns `True` if the permutation ``p`` is a perfect elimination order 
    for a Cholesky factorization of ``X``.

    :param X: CVXOPT sparse square matrix of doubles.  Only the sparsity
              pattern of the lower triangular part is accessed
    :param p: CVXOPT dense integer matrix of length `n`, if `n` is the order
              of ``X``


.. function:: perm(X, p)

    Performs a symmetric permutation of a square sparse matrix.

    :samp:`Y = perm(X, p)`

    This is equivalent to but more efficient than :samp:`Y = X[p, p]`.

    :param X: CVXOPT sparse square matrix of doubles
    :param p: CVXOPT dense integer matrix of length ``n``, if ``n`` is the 
        order of ``X``
    :returns: CVXOPT sparse square matrix of doubles


.. function:: symmetrize(X)

    Symmetrizes a lower triangular matrix.  Returns 
  
    .. math::

        Y := X + X^T - \mathrm{diag}(\mathrm{diag}(X)) 

    where :math:`X` is a lower triangular matrix.

    :param X: CVXOPT sparse square matrix of doubles.  Must be lower 
        triangular
    :returns: CVXOPT sparse square matrix of doubles


.. function:: tril(X)

    Returns the lower triangular part of a sparse matrix ``X``.

    :param X: CVXOPT sparse square matrix of doubles
    :returns: CVXOPT sparse square matrix of doubles

.. function:: maxchord(X, k)

   Computes maximal chordal subgraph of sparsity graph and returns the
   projection of X on the chordal subgraph as a chordal matrix. The
   optional parameter `k` determines the last vertex in a perfect
   elimination ordering of the maximal chordal subgraph. A node of
   maximum degree is chosen if `k` is not specified.

    :param X: CVXOPT sparse square matrix of doubles.  Only the sparsity
        pattern of the lower triangular part of the matrix is
        accessed.
    :param k: integer between :math:`0` and :math:`n-1` if :math:`n`
        is the order of ``X``

    :returns: chordal matrix
