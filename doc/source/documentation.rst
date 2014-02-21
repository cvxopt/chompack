Documentation
===============

.. module:: chompack
   :synopsis: Library for chordal matrix computations.
.. moduleauthor:: Martin S. Andersen <martin.skovgaard.andersen@gmail.com>
.. moduleauthor:: Lieven Vandenberghe <vandenbe@ee.ucla.edu>

Quick start
--------------

The core functionality of CHOMPACK is contained in two types of
objects: the :py:class:`symbolic` object and the :py:class:`cspmatrix`
(chordal sparse matrix) object. A :py:class:`symbolic` object
represents a symbolic factorization of a sparse symmetric matrix
:math:`A`, and it can be created as follows:

.. code-block:: python

	from cvxopt import spmatrix, amd
	import chompack as cp

	# generate sparse matrix
	I = [0, 1, 3, 1, 5, 2, 6, 3, 4, 5, 4, 5, 6, 5, 6]
	J = [0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6]
	A = spmatrix(1.0, I, J, (7,7))

	# compute symbolic factorization using AMD ordering
	symb = cp.symbolic(A, p=amd.order)

The argument :math:`p` is a so-called elimination order, and it can
be either an ordering routine or a permutation vector. In the above
example we used the "approximate minimum degree" (AMD) ordering
routine. Note that :math:`A` is a lower-triangular sparse matrix that represents a
symmetric matrix; upper-triangular entries in :math:`A` are ignored in the
symbolic factorization.

Now let's inspect the sparsity pattern of `A` and its chordal
embedding (i.e., the filled pattern):

.. code-block:: python

	>>> print(A)
	[ 1.00e+00     0         0         0         0         0         0    ]
	[ 1.00e+00  1.00e+00     0         0         0         0         0    ]
	[    0         0      1.00e+00     0         0         0         0    ]
	[ 1.00e+00     0         0      1.00e+00     0         0         0    ]
	[    0         0         0      1.00e+00  1.00e+00     0         0    ]
	[    0      1.00e+00     0      1.00e+00  1.00e+00  1.00e+00     0    ]
	[    0         0      1.00e+00     0      1.00e+00     0      1.00e+00]

	>>> print(symb.sparsity_pattern(reordered=False, symmetric=False))
	[ 1.00e+00     0         0         0         0         0         0    ]
	[ 1.00e+00  1.00e+00     0         0         0         0         0    ]
	[    0         0      1.00e+00     0         0         0         0    ]
	[ 1.00e+00     0         0      1.00e+00     0         0         0    ]
	[    0         0         0      1.00e+00  1.00e+00     0         0    ]
	[ 1.00e+00  1.00e+00     0      1.00e+00  1.00e+00  1.00e+00     0    ]
	[    0         0      1.00e+00     0      1.00e+00     0      1.00e+00]

The reordered pattern and its cliques can be inspected using the
following commands:

.. code-block:: python

	>>> print(symb)
	[X X          ]
	[X X X        ]
	[  X X X   X  ]
	[    X X   X X]
	[        X X X]
	[    X X X X X]
	[      X X X X]

	>>> print(symb.cliques())
	[[0, 1], [1, 2], [2, 3, 5], [3, 5, 6], [4, 5, 6]]

Similarly, the clique tree, the supernodes, and the separator sets are:

.. code-block:: python

	>>> print(symb.parent())
	[1, 2, 3, -1, 3]

	>>> print(symb.supernodes())
	[[0], [1], [2], [3, 5, 6], [4]]	
	
	>>> print(symb.separators())
	[[1], [2], [3, 5], [], [5, 6]]

The :py:class:`cspmatrix` object represents a chordal sparse matrix,
and it contains lower-triangular numerical values as well as a
reference to a symbolic factorization that defines the sparsity
pattern. Given a :py:class:`symbolic` object `symb` and a sparse
matrix :math:`A`, we can create a :py:class:`cspmatrix` as follows:

.. code-block:: python

	from cvxopt import spmatrix, amd, printing
	import chompack as cp
	printing.options['dformat'] = '%3.1f'

	# generate sparse matrix and compute symbolic factorization
	I = [0, 1, 3, 1, 5, 2, 6, 3, 4, 5, 4, 5, 6, 5, 6]
	J = [0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6]
	A = spmatrix([1.0*i for i in range(15)], I, J, (7,7))
	symb = cp.symbolic(A, p=amd.order)

	L = cp.cspmatrix(symb)
	L += A
	
Now let us take a look at  :math:`A` and :math:`L`:

.. code-block:: python

	>>> print(A)
	[ 0.0  0    0    0    0    0    0  ]
	[ 1.0  3.0  0    0    0    0    0  ]
	[ 0    0    5.0  0    0    0    0  ]
	[ 2.0  0    0    7.0  0    0    0  ]
	[ 0    0    0    8.0 10.0  0    0  ]
	[ 0    4.0  0    9.0 11.0 13.0  0  ]
	[ 0    0    6.0  0   12.0  0   14.0]

	>>> print(L)
	[ 5.0  0    0    0    0    0    0  ]
	[ 6.0 14.0  0    0    0    0    0  ]
	[ 0   12.0 10.0  0    0    0    0  ]
	[ 0    0    8.0  7.0  0    0    0  ]
	[ 0    0    0    0    3.0  0    0  ]
	[ 0    0   11.0  9.0  4.0 13.0  0  ]
	[ 0    0    0    2.0  1.0  0.0  0.0]

Notice that :math:`L` is a reordered lower-triangular representation
of :math:`A`. We can convert :math:`L` to an :py:class:`spmatrix` using
the `spmatrix()` method:

.. code-block:: python

	>>> print(L.spmatrix(reordered = False))
	[ 0.0  0    0    0    0    0    0  ]
	[ 1.0  3.0  0    0    0    0    0  ]
	[ 0    0    5.0  0    0    0    0  ]
	[ 2.0  0    0    7.0  0    0    0  ]
	[ 0    0    0    8.0 10.0  0    0  ]
	[ 0.0  4.0  0    9.0 11.0 13.0  0  ]
	[ 0    0    6.0  0   12.0  0   14.0]

Notice that this returns an :py:class:`spmatrix` with the same ordering
as :math:`A`, i.e., the inverse permutation is applied to :math:`L`.  

The following example illustrates how to use the Cholesky routine:

.. code-block:: python

   from cvxopt import spmatrix, amd, normal
   from chompack import symbolic, cspmatrix, cholesky

   # generate sparse matrix and compute symbolic factorization
   I = [0, 1, 3, 1, 5, 2, 6, 3, 4, 5, 4, 5, 6, 5, 6]
   J = [0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6]
   A = spmatrix([0.1*(i+1) for i in range(15)], I, J, (7,7)) + spmatrix(10.0,range(7),range(7))
   symb = symbolic(A, p=amd.order)
   
   # create cspmatrix 
   L = cspmatrix(symb)
   L += A 

   # compute numeric factorization
   cholesky(L)

.. code-block:: python 

   >>> print(L)
   [ 3.26e+00     0         0         0         0         0         0    ]
   [ 2.15e-01  3.38e+00     0         0         0         0         0    ]
   [    0      3.84e-01  3.31e+00     0         0         0         0    ]
   [    0         0      2.72e-01  3.28e+00     0         0         0    ]
   [    0         0         0         0      3.22e+00     0         0    ]
   [    0         0      3.63e-01  2.75e-01  1.55e-01  3.34e+00     0    ]
   [    0         0         0      9.16e-02  6.20e-02 -1.04e-02  3.18e+00]



.. code-block:: python

   from cvxopt import spmatrix, printing
   printing.options['width'] = -1
   import chompack as cp
   
   # Define chordal sparse matrix
   I = range(17)+[2,2,3,3,4,14,4,14,8,14,15,8,15,7,8,14,8,14,14,\
		15,10,12,13,16,12,13,16,12,13,15,16,13,15,16,15,16,15,16,16]
   J = range(17)+[0,1,1,2,2,2,3,3,4,4,4,5,5,6,6,6,7,7,8,\
		8,9,9,9,9,10,10,10,11,11,11,11,12,12,12,13,13,14,14,15]
   A = spmatrix(1.0,I,J,(17,17))

   # Compute maximum cardinality search 
   p = cp.maxcardsearch(A)

.. code-block:: python

   >>> cp.peo(A,p)
   True
   >>> print(list(p))
   [0, 9, 10, 11, 12, 13, 5, 6, 7, 1, 2, 3, 4, 8, 14, 15, 16]
   >>> symb = cp.symbolic(A,p)
   >>> print(symb)
   [X                   X            ]
   [  X X   X X                     X]
   [  X X   X X                     X]
   [      X X X                   X X]
   [  X X X X X                   X X]
   [  X X X X X                   X X]
   [            X             X   X  ]
   [              X X         X X    ]
   [              X X         X X    ]
   [                  X X X          ]
   [X                 X X X X   X    ]
   [                  X X X X   X    ]
   [                    X X X X X X  ]
   [            X X X       X X X X  ]
   [              X X   X X X X X X X]
   [      X X X X           X X X X X]
   [  X X X X X                 X X X]
   >>> print(symb.fill)
   (0, 0)



Symbolic factorization
-----------------------

.. autoclass:: chompack.symbolic
   :members:


Chordal sparse matrices
------------------------

.. autoclass:: chompack.cspmatrix
   :members: 


Numerical computations
----------------------

.. autofunction:: chompack.cholesky

.. autofunction:: chompack.llt

.. autofunction:: chompack.projected_inverse

.. autofunction:: chompack.completion

.. autofunction:: chompack.hessian

.. autofunction:: chompack.dot

.. autofunction:: chompack.trmm

.. autofunction:: chompack.trsm



Chordal conversion
------------------
The following example illustrates how to apply the chordal conversion
technique to a sparse SDP.

.. code-block:: python

   # Given: tuple with cone LP problem data
   # prob = (c,G,h,dims,A,b)

   # Solve cone LP with CVXOPT's conelp() routine
   sol = cvxopt.conelp(*prob)

   # Apply chordal conversion to cone LP 
   probc, blk2sparse, symbs = chompack.convert_conelp(*prob)

   # Solve converted problem with CVXOPT's conelp() routine
   solc = cvxopt.solvers.conelp(*probc)
   

.. autofunction:: chompack.convert_conelp

.. autofunction:: chompack.convert_block


Auxiliary routines
-------------------

.. autofunction:: chompack.maxcardsearch

.. autofunction:: chompack.peo

.. autofunction:: chompack.maxchord

.. autofunction:: chompack.merge_size_fill

.. autofunction:: chompack.tril

.. autofunction:: chompack.triu

.. autofunction:: chompack.perm

.. autofunction:: chompack.symmetrize

