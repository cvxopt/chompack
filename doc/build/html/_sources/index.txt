Chordal Matrix Package—a library for chordal matrix computations
======================================================================

Chompack is a library for chordal matrix computations.  It includes
routines for:

  - symbolic factorization
  - numeric Cholesky factorization
  - forward and back substitution
  - maximum determinant positive definite completion
  - computations with logarithmic barriers for sparse matrix cones
  - chordal conversion
  - computing maximal chordal subgraph

The implementation is based on the supernodal-multifrontal algorithms
described in this paper:

.. seealso::
     
      M. S. Andersen, J. Dahl, and L. Vandenberghe, `Logarithmic barriers
      for sparse matrix cones <http://arxiv.org/abs/1203.2742>`_, *Optimization Methods and Software*, 2013.

Applications of these algorithms in optimization include sparse matrix cone programs,
covariance selection, graphical models, and decomposition and relaxation methods.


.. toctree::
   :hidden:
      
   license
   install
   documentation
