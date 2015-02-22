
Quick start
-----------

                
The core functionality of CHOMPACK is contained in two types of
objects: the :py:class:`symbolic` object and the :py:class:`cspmatrix`
(chordal sparse matrix) object. A :py:class:`symbolic` object
represents a symbolic factorization of a sparse symmetric matrix
:math:`A`, and it can be created as follows:

                
.. code:: python

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

                
.. code:: python

    >>> print(A)

.. parsed-literal::

    [ 1.00e+00     0         0         0         0         0         0    ]
    [ 1.00e+00  1.00e+00     0         0         0         0         0    ]
    [    0         0      1.00e+00     0         0         0         0    ]
    [ 1.00e+00     0         0      1.00e+00     0         0         0    ]
    [    0         0         0      1.00e+00  1.00e+00     0         0    ]
    [    0      1.00e+00     0      1.00e+00  1.00e+00  1.00e+00     0    ]
    [    0         0      1.00e+00     0      1.00e+00     0      1.00e+00]
    


.. code:: python

    >>> print(symb.sparsity_pattern(reordered=False, symmetric=False))

.. parsed-literal::

    [ 1.00e+00     0         0         0         0         0         0    ]
    [ 1.00e+00  1.00e+00     0         0         0         0         0    ]
    [    0         0      1.00e+00     0         0         0         0    ]
    [ 1.00e+00     0         0      1.00e+00     0         0         0    ]
    [    0         0         0      1.00e+00  1.00e+00     0         0    ]
    [ 1.00e+00  1.00e+00     0      1.00e+00  1.00e+00  1.00e+00     0    ]
    [    0         0      1.00e+00     0      1.00e+00     0      1.00e+00]
    


                
The reordered pattern and its cliques can be inspected using the
following commands:

                
.. code:: python

    >>> print(symb)

.. parsed-literal::

    [X X          ]
    [X X X        ]
    [  X X   X X  ]
    [      X   X X]
    [    X   X X X]
    [    X X X X X]
    [      X X X X]
    


.. code:: python

    >>> print(symb.cliques())

.. parsed-literal::

    [[0, 1], [1, 2], [2, 4, 5], [3, 5, 6], [4, 5, 6]]


                
Similarly, the clique tree, the supernodes, and the separator sets are:

                
.. code:: python

    >>> print(symb.parent())

.. parsed-literal::

    [1, 2, 4, 4, 4]


.. code:: python

    >>> print(symb.supernodes())

.. parsed-literal::

    [[0], [1], [2], [3], [4, 5, 6]]


.. code:: python

    >>> print(symb.separators())

.. parsed-literal::

    [[1], [2], [4, 5], [5, 6], []]


                
The :py:class:`cspmatrix` object represents a chordal sparse matrix,
and it contains lower-triangular numerical values as well as a
reference to a symbolic factorization that defines the sparsity
pattern. Given a :py:class:`symbolic` object `symb` and a sparse
matrix :math:`A`, we can create a :py:class:`cspmatrix` as follows:

                
.. code:: python

    from cvxopt import spmatrix, amd, printing
    import chompack as cp
    printing.options['dformat'] = '%3.1f'
    
    # generate sparse matrix and compute symbolic factorization
    I = [0, 1, 3, 1, 5, 2, 6, 3, 4, 5, 4, 5, 6, 5, 6]
    J = [0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6]
    A = spmatrix([1.0*i for i in range(1,15+1)], I, J, (7,7))
    symb = cp.symbolic(A, p=amd.order)
    
    L = cp.cspmatrix(symb)
    L += A
                
Now let us take a look at  :math:`A` and :math:`L`:

                
.. code:: python

    >>> print(A)

.. parsed-literal::

    [ 1.0  0    0    0    0    0    0  ]
    [ 2.0  4.0  0    0    0    0    0  ]
    [ 0    0    6.0  0    0    0    0  ]
    [ 3.0  0    0    8.0  0    0    0  ]
    [ 0    0    0    9.0 11.0  0    0  ]
    [ 0    5.0  0   10.0 12.0 14.0  0  ]
    [ 0    0    7.0  0   13.0  0   15.0]
    


.. code:: python

    >>> print(L)

.. parsed-literal::

    [ 6.0  0    0    0    0    0    0  ]
    [ 7.0 15.0  0    0    0    0    0  ]
    [ 0   13.0 11.0  0    0    0    0  ]
    [ 0    0    0    4.0  0    0    0  ]
    [ 0    0    9.0  0    8.0  0    0  ]
    [ 0    0   12.0  5.0 10.0 14.0  0  ]
    [ 0    0    0    2.0  3.0  0.0  1.0]
    


                
Notice that :math:`L` is a reordered lower-triangular representation
of :math:`A`. We can convert :math:`L` to an :py:class:`spmatrix` using
the `spmatrix()` method:

                
.. code:: python

    >>> print(L.spmatrix(reordered = False))

.. parsed-literal::

    [ 1.0  0    0    0    0    0    0  ]
    [ 2.0  4.0  0    0    0    0    0  ]
    [ 0    0    6.0  0    0    0    0  ]
    [ 3.0  0    0    8.0  0    0    0  ]
    [ 0    0    0    9.0 11.0  0    0  ]
    [ 0.0  5.0  0   10.0 12.0 14.0  0  ]
    [ 0    0    7.0  0   13.0  0   15.0]
    


                
This returns an :py:class:`spmatrix` with the same ordering
as :math:`A`, i.e., the inverse permutation is applied to :math:`L`.

The following example illustrates how to use the Cholesky routine:

                
.. code:: python

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
.. code:: python

    >>> print(L)

.. parsed-literal::

    [ 3.3  0    0    0    0    0    0  ]
    [ 0.2  3.4  0    0    0    0    0  ]
    [ 0    0.4  3.3  0    0    0    0  ]
    [ 0    0    0    3.2  0    0    0  ]
    [ 0    0    0.3  0    3.3  0    0  ]
    [ 0    0    0.4  0.2  0.3  3.3  0  ]
    [ 0    0    0    0.1  0.1 -0.0  3.2]
    


                
Given a sparse matrix :math:`A`, we can check if it is chordal by 
checking whether the permutation :math:`p` returned by maximum cardinality 
search is a perfect elimination ordering:

                
.. code:: python

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
                
Is :math:`p` a perfect elimination ordering?

                
.. code:: python

    >>> cp.peo(A,p)



.. parsed-literal::

    True



                
Let's verify that no fill is generated by the symbolic factorization:

                
.. code:: python

    >>> symb = cp.symbolic(A,p)
    >>> print(symb.fill)

.. parsed-literal::

    (0, 0)

