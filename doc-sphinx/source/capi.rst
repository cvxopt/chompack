.. _c-capi:

***********
C Interface
***********

The CHOMPACK routines are available as a standard C library.


Creating chordal matrices
=========================

The two most basic data-structures in CHOMPACK are a :c:type:`cliqueforest`, which represents  
the sparsity pattern of symmetric matrix, and a :c:type:`chordalmatrix` which attaches
numerical values to a sparsity pattern defined by a :c:type:`cliqueforest`. 

A clique-forest is created using the :c:func:`cliqueforest_create` function.

.. c:function:: int cliqueforest_create(cliqueforest **F, const ccs *X, const int_t *p)

    Creates a :c:type:`cliqueforest` from a :c:type:`ccs` matrix.  
        
    :param F: is overwritten with a pointer to the created :c:type:`cliqueforest` object. 
    :param X: ccs matrix. `X` must be chordal (e.g., created by :c:func:`chordalembedding`).
    :param p: permutation associated with `X` of order `n`, if `n` is the order of `X`. 
        Typically, `p` is a fill-in reducing permutation used for creating `X` with
        :c:func:`chordalembedding`. 
    :returns: a status integer `CHOMPACK_OK`, or `CHOMPACK_NOMEMORY`. 

A clique-forest is destroyed using the function:

.. c:function:: void cliqueforest_destroy(cliqueforest *F)

We can create a :c:type:`chordalmatrix` for a particular :c:type:`cliqueforest` using
the routine:

.. c:function:: chordalmatrix * ccs_to_chordalmatrix (const cliqueforest *F, const ccs *X) 

    Creates a chordal matrix by projecting a sparse matrix `X` onto the sparsity pattern
    defined by the clique-forest `F`. Elements in `X` outside the chordal sparsity pattern
    in `F` are ignored.
    
    :param F: a :c:type:`cliqueforest` object specifying a chordal sparsity pattern.
    :param X: a sparse :c:type:`ccs` matrix with the numerical values.
    :returns: a chordal matrix object, or `NULL` if a memory error occured.  

and a :c:type:`chordalmatrix` object is destroyed using the routine:

.. c:function:: void chordalmatrix_destroy(chordalmatrix *X);

Internally, CHOMPACK does not distinguish between chordal matrices and factors; the
user of the library must make that distinction. In the following we will refer to
a symmetric :c:type:`chordalmatrix` object as a `chordal matrix` and a factor 
stored in a :c:type:`chordalmatrix` as a `chordal factor`.

A chordal matrix is converted to
:c:type:`ccs` format using the routine:

.. c:function:: ccs * chordalmatrix_to_ccs (const chordalmatrix *A)

and a chordal factor is converted using the routine:
 
.. c:function:: ccs * chordalfactor_to_ccs (const chordalmatrix *A)

Finally, a copy of a :c:type:`chordalmatrix` is created using the routine:

.. c:function:: chordalmatrix * chordalmatrix_copy(const chordalmatrix *A)

Computational routines
======================
The CHOMPACK library provides the following computational routines.

.. c:function:: int cholesky(chordalmatrix *X)

    Computes a zero fill-in Cholesky factorization 

    .. math:: 
        :label: e-chol
        
        P^TXP = LL^T

    of a positive definite chordal matrix ``X``. 

    :param X: a chordal matrix. On entry, ``X`` contains a positive definite chordal matrix,
        and on exit it is overwritten with its Cholesky factor.
        
    :returns: `CHOMPACK_OK` if the factorization was successful,  
        `CHOMPACK_FACTORIZATION_ERR` if the factorization failed, or
        `CHOMPACK_NOMEMORY`.
    
.. c:function:: int solve(const chordalmatrix *L, double *X, int m, int sys)

    Solves a factored set of equations, or multiplies with Cholesky factors.

    ``L`` contains the factors of a factorization :eq:`e-chol` of a 
    positive definite sparse chordal matrix.  ``X`` is a dense matrix
    of doubles with the same number of rows as ``L``, stored as a
    contigious array on column-major-order. 
    On exit, ``X`` is overwritten with one of the four matrices in the 
    table.

    .. tabularcolumns:: |l|c|
    
    =======================   ====
    action                    sys
    =======================   ====
    :math:`X := L^{-1}P^TX`   0
    :math:`X := PL^{-T}X`     1
    :math:`X := PLX`          2
    :math:`X := L^TP^TX`      3
    =======================   ====

    :param L: chordal factor
    :param X: dense matrix of doubles stored in columm-major-order with `n` rows, if `n` is the order of `L`
    :param m: number of columns of ``X``
    :param sys: integer

.. c:function:: chordalmatrix * completion(const chordalmatrix *X, int *info)

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
    :param info: status integer overwritten with
        `CHOMPACK_OK`, `CHOMPACK_FACTORIZATION_ERR` or `CHOMPACK_NOMEMORY`
    :returns: chordal factor with the same sparsity pattern as `X`    

.. c:function:: int partial_inv(chordalmatrix *L)

    Evaluates the projection of the inverse of the matrix 
    :math:`X = PLL^TP^T` on the sparsity pattern of :math:`X`.

    Overwrites ``L`` with lower-triangular part of  

    .. math::

        Y = \mathrm{proj}(X^{-1}) 

    where :math:`X` is a positive definite chordal matrix specified by
    its Cholesky factor :math:`P^TXP = LL^T`.

    :param L: chordal factor
    :returns: `CHOMPACK_OK`, `CHOMPACK_FACTORIZATION_ERR` or `CHOMPACK_NOMEMORY`

.. c:function:: int hessian_factor(const chordalmatrix *L, const chordalmatrix *Y, chordalmatrix **U, int adj, int inv, int m)

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

    The :c:func:`hessian_factor` function
    evaluates these mappings or their inverses for a list of symmetric 
    chordal matrices ``**U``, and overwrites the
    matrices with the results.   The following table lists the possible 
    actions.
    
    .. tabularcolumns:: |L|L|L|   FIXME: This has no effect, opposed to Sphinx doc.
    
    ================================================  ===  ===
    Action                                            adj  inv
    ================================================  ===  ===
    :math:`U_i:={\cal G}_X(U_i)`                       0    0
    :math:`U_i:={\cal G}_X^{-1}(U_i)`                  0    1
    :math:`U_i:={\cal G}_X^\mathrm{adj}(U_i)`          1    0
    :math:`U_i:=({\cal G}_X^\mathrm{adj})^{-1}(U_i)`   1    1  
    ================================================  ===  ===
    
    The input argument ``L`` is the Cholesky factor of :math:`X`,  
    as computed by the :c:func:`cholesky` function.
    The input argument ``Y`` is the partial inverse of the inverse of
    :math:`X`, as computed by the :c:func:`partial_inv` function.
    The input argument ``U`` is a list of CHOMPACK matrices with the 
    same sparsity pattern as ``L`` and ``Y``.

    The matrices :math:`\mathcal H_X(U_i)` can be computed by two calls

    .. code-block:: c

        hessian_factor(L, Y, U, 0, 0);
        hessian_factor(L, Y, U, 1, 0);

    The matrices :math:`\mathcal H_X^{-1}(U_i)` can be computed as

    .. code-block:: c

        hessian_factor(L, Y, U, 1, 1);
        hessian_factor(L, Y, U, 0, 1);
        
    :param L: chordal factor
    :param Y: chordal matrix with the same sparsity pattern as ``L``.
    :param U: list of CHOMPACK matrices with the same sparsity pattern
        as ``L`` and ``Y``
    :param adj: 0/1
    :param inv: 0/1


Auxiliary routines 
==================

.. c:function:: void scal(double a, chordalmatrix *X)

    Evaluates 

    .. math:: 
    
        X := \alpha X.
 
    :param alpha: scaling factor
    :param X: chordal matrix


.. c:function:: void axpy(const chordalmatrix *X, chordalmatrix *Y, double a)

    Evaluates 

    .. math::

        Y := \alpha X + Y.
    
    :param X: chordal matrix
    :param Y: chordal matrix with the same sparsity pattern as ``X``
    :param alpha: float

.. c:function:: int llt(chordalmatrix *L)

    On entry `L` contains a Cholesky factor of :math:`P^T X P = L L^T`.
    On exit, `L` is overwritten with `X`.
    
    :returns: `CHOMPACK_OK` or `CHOMPACK_NOMEMORY`    
        
.. c:function:: double dot(const chordalmatrix *X, const chordalmatrix *Y)

    Returns the inner product 
    
    .. math::

        \mathrm{tr}(XY)

    of two symmetric sparse matrices with the same chordal sparsity pattern.
    
    :param X: chordal matrix
    :param Y: chordal matrix with have the same sparsity pattern as ``X``

.. c:function:: double logdet(const chordalmatrix *L)

    Returns the logarithm of the determinant of a Cholesky factor ``L``.

    :param L: chordal factor
    

Routines for sparse CCS matrices
================================

Sparse matrices are specified in `compressed-column-storage` using the (:c:type:`ccs`) 
data-structure. For a general `nrows` by `ncols` sparse matrix with `nnz` nonzero 
entries this means the following.  The sparsity pattern and the nonzero values are 
stored in three fields:

:c:member:`values` 
    A :c:type:`double` array with the 
    nonzero entries of the matrix stored columnwise.  

:c:member:`rowind` 
    An array of integers of length `nnz` containing the row indices of 
    the nonzero entries sorted in increasing order, stored in the same 
    order as :c:member:`values`.

:c:member:`colptr` 
    An array of integers of length `ncols` + 1 with for each column of the 
    matrix the index of the first element in :c:member:`values` from that 
    column.  More precisely, ``colptr[0]`` is :const:`0`, and for 
    k = 0, 1, ..., `ncols` - 1, ``colptr[k+1]`` is equal to 
    ``colptr[k]`` plus the number of nonzeros in column `k` of the
    matrix.  Thus, ``colptr[ncols]`` is equal to `nnz`, the number of 
    nonzero entries.


For example, for the matrix

.. math::

    A=\left [\begin{array}{cccc}
        1 & 0 & 0 & 5\\
        2 & 0 & 4 & 0\\
        0 & 0 & 0 & 6\\
        3 & 0 & 0 & 0
    \end{array}\right]

the elements of :c:member:`values`, :c:member:`rowind`, and :c:member:`colptr` 
are:

:c:member:`values`:
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0

:c:member:`rowind`:
    0, 1,3, 1, 0, 2

:c:member:`colptr`: 
    0, 3, 3, 4, 6.


A :c:type:`ccs` data-structure can be allocated using the routine:

.. c:function:: ccs * alloc_ccs(int_t nrows, int_t ncols, int nnz)

    Allocates a `nrows` times `ncols` :c:type:`ccs` matrix with `nnz` non-zero elements.

and a :c:type:`ccs` structure can be freed with the routine:

.. c:function:: void free_ccs(ccs *obj)

A given :c:type:`ccs` can be embedded into a chordal sparsity pattern (i.e., it can 
be triangulated) using the routine:

.. c:function:: ccs * chordalembedding(const ccs *X, int *nfill)

    :param X:   a :c:type:`ccs` matrix to be triangulated.
    :param nfill:  an integer that will be overwritten with the amount of fill-in 
        (i.e., the number of edges added in the triangulation process).

    :returns:  a chordal embedding of `X` as a :c:type:`ccs` matrix.

Additionally,  a number of auxilliary routines for :c:type:`ccs` matrices are available.

.. c:function:: ccs * perm(ccs *A, int_t *p)

    Returns a symmetric permutation :math:`P^TAP` where
    `P` is a permutation and `A` is a square matrix of dimension `n`.
    
    :param A: :c:type:`ccs` matrix.
    :param p: a valid permutation of order `n`, if `n` is the dimension of `A`.
    :return: a symmetric permutation of the rows and columns of `A`.
            
.. c:function:: ccs * symmetrize(const ccs *A)

    Symmetrizes a lower triangular matrix `A`, i.e., computes 
    
    .. math::
        A+A^T-\mathbf{diag}(\mathbf{diag}(A))
    
    :param A: a lower triangular matrix.
    :returns: a symmetric matrix, with `A` as lower triangular part.
     
.. c:function:: ccs * tril(const ccs *A)

    Returns the lower-triangular part of a sparse matrix.
    
    :param A: :c:type:`ccs` matrix.
    :returns: a :c:type:`ccs` matrix, with `A` as lower triangular part, and zeros
        elsewhere.
        
Overview of data-structures
===========================

.. c:type:: ccs
   
    .. code-block:: c
    
        typedef struct {
            void  *values;      /* value list */
            int_t *colptr;      /* column pointer list */
            int_t *rowind;      /* row index list */
            int_t nrows, ncols; /* number of rows and columns */
            int   id;           /* not currently used */
        } ccs;

    .. c:member:: nrows
    
        number of rows

    .. c:member:: ncols
    
        number of columns
        
    .. c:member:: values
    
        length `colptr[ncols]` :c:type:`double` array with numerical values.
        
    .. c:member:: colptr
    
        length `ncols+1` array with compressed column-indices.
        
    .. c:member:: rowind
   
        length `colptr[ncols]` array with row-indices.
    
    .. c:member:: id
    
        included for compatibility with CVXOPT; not currently used.    
            
.. c:type:: cliqueforest

    .. code-block:: c

        typedef struct cliqueforest {
        
          int nRoots, nCliques, n;
          struct clique **roots, **list, *location;
          int_t *p, *ip;
        
        } cliqueforest;

    .. c:member:: nRoots
    
        number of clique-trees in the clique-forest

    .. c:member:: nCliques
    
        number of cliques in the clique-forest

    .. c:member:: n 
    
        number of nodes (dimension of the matrix)

    .. c:member:: roots
    
        array of pointers to each root element

    .. c:member:: list
    
        array of pointers to each clique numbers in reverse topological order

    .. c:member:: p
    
        permutation used for creating the clique-forst

    .. c:member:: ip
    
        inverse permutation 

    .. c:member:: location
    
        used internally by CHOMPACK

.. c:type:: chordalmatrix

    .. code-block:: c

        typedef struct {
        
          cliqueforest *F;
          double **US;
          double **SS;
          double **UU;
          void *F_py;
        
        } chordalmatrix;

    .. c:member:: F
    
        clique-forest representing sparsity pattern of matrix

    .. c:member:: US
    
        array of matrices corresponding the :math:`\{U_k,S_k\}` blocks of the chordal matrix

    .. c:member:: SS
    
        array of matrices corresponding the :math:`\{S_k,S_k\}` blocks of the chordal matrix

    .. c:member:: UU
    
        array of matrices corresponding the :math:`\{U_k,U_k\}` blocks of the chordal matrix
        
    .. c:member:: F_py
    
        included for compatibility with CVXOPT

Examples
========

The following example computes the partial inverse of the chordal matrix

.. math:: 

    A=\left [\begin{array}{ccccc}
        1 & 0 & 0 & 0 & 1\\
        0 & 10 & 0 & 0 & 1\\
        0 & 0 & 10 & 0 & 1\\
        0 & 0 & 0 & 10 & 1\\
        1 & 1 & 1 & 1 & 10        
    \end{array}\right]

.. code-block:: c
 
    #include <stdio.h>
    #include "chompack.h"
    
    int main() {
    
      int j, k;
      double Aval[] = {10.0, 1.0, 10.0, 1.0, 10.0, 1.0, 10.0, 1.0, 10.0};
      int_t  Acol[] = {0, 2, 4, 6, 8, 9};
      int_t  Arow[] = {0, 4, 1, 4, 2, 4, 3, 4, 5};
      ccs    A = { .values = Aval, .colptr = Acol, .rowind = Arow, .nrows = 5, .ncols = 5, 0 };
    
      printf("creating cliquetree\n");
      cliqueforest *F;
      cliqueforest_create(&F, &A, NULL);
    
      printf("creating chordal matrix\n");
      chordalmatrix *X = ccs_to_chordalmatrix (F, &A);
    
      printf("performing factorization\n");
      cholesky(X);
    
      printf("computing partial inverse\n");
      partial_inv(X);
    
      ccs *Y = chordalmatrix_to_ccs(X);
    
      printf("partial inverse:\n");
      for (j=0; j<Y->ncols; j++)
        for (k=Y->colptr[j]; k<Y->colptr[j+1]; k++)
          printf("(%i,%i): % 3.2e\n", (int)Y->rowind[k], j, ((double *)Y->values)[k]);
    
      chordalmatrix_destroy(X);
      cliqueforest_destroy(F);
      free_ccs(Y);
    
      return 0;
    
    }
 
