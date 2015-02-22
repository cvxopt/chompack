Documentation
===============

.. module:: chompack
   :synopsis: Library for chordal matrix computations.
.. moduleauthor:: Martin S. Andersen <martin.skovgaard.andersen@gmail.com>
.. moduleauthor:: Lieven Vandenberghe <vandenbe@ee.ucla.edu>


.. include:: quick_start.rst
    

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

.. autofunction:: chompack.psdcompletion

.. autofunction:: chompack.edmcompletion
		  
.. autofunction:: chompack.hessian

.. autofunction:: chompack.dot

.. autofunction:: chompack.trmm

.. autofunction:: chompack.trsm

.. autoclass:: chompack.pfcholesky
   :members: 




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

.. autofunction:: chompack.pybase.plot.spy

.. autofunction:: chompack.pybase.plot.sparsity_graph

.. autofunction:: chompack.pybase.plot.etree_graph
