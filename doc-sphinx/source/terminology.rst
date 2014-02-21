.. _terminology:

***********
Terminology 
***********

The following notation and terminology is used in this documentation.

A sparsity pattern of a square matrix is a set of positions ``(i,j)`` 
where the matrix is zero.  A sparsity pattern is symmetric if ``(j,i)``
is in the sparsity pattern whenever ``(i,j)`` is in the sparsity pattern.
Note that the entries of the matrix outside the sparsity pattern are 
allowed to be zero.  Those entries are called *numerical zeros*, as opposed
to the *structural zeros* in the sparsity pattern.

A symmetric sparsity pattern is called *chordal* if every positive definite 
matrix :math:`A` with the sparsity pattern can be factored as

.. math:: 

   P^T A P  = L L^T  

where :math:`P` is a permutation matrix, and :math:`L` is a lower 
triangular matrix with the same sparsity pattern as the lower triangular 
part of :math:`P^T A P`.  In other words, there exists a symmetric 
reordering of the matrix that has a Cholesky factorization with zero
fill-in.  Such a reordering is called a *perfect elimination ordering* for 
the sparsity pattern.  

For simplicity, we will refer to a symmetric matrix with a chordal sparsity 
pattern as a *chordal symmetric matrix*.  This terminology is somewhat 
ambiguous, because there can be many sparsity patterns associated with the
same matrix.  When we use the term it will be clear from the context
which sparsity pattern is used.  A nonsymmetric matrix :math:`X` is chordal
if the sparsity pattern of :math:`X + X^T` is chordal.

The *projection* of a matrix on a sparsity pattern ``S`` is defined as

.. math::

    \mathrm{proj}(X)_{ij} = 
    \left\{ \begin{array}{ll}
        0       & (i,j) \in S \\
        X_{ij}  & (i,j) \not\in S.
    \end{array}\right.

A sparsity pattern ``R`` is an *embedding* of the sparsity pattern ``S`` if
``R``  is a subset of ``S``. 
