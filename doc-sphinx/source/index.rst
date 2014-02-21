########
CHOMPACK 
########

CHOMPACK is a library of algorithms for matrix computations with 
chordal sparsity patterns. 
It includes routines for Cholesky factorization and maximum determinant 
positive definite completion of chordal matrices, 
evaluation of the gradient, Hessian, and inverse Hessian of the 
logarithmic barrier function of a cone of positive definite matrices with 
chordal sparsity pattern, 
and evaluation of gradient, Hessian, and inverse Hessian of the conjugate 
barrier.

The library provides efficient multifrontal implementations of the 
algorithms in the paper
`Covariance selection for non-chordal graphs via chordal embedding 
<http://www.ee.ucla.edu/~vandenbe/publications/covsel.pdf>`_
by J. Dahl, L. Vandenberghe, V. Roychowdhury  
(Optimization Methods and Software 23 (4), 501-520, 2008).

.. toctree::
    :hidden:
    
    copyright.rst
    terminology.rst
    pythonapi.rst
    pythonexamples.rst 
    capi.rst
    installation.rst
    download.rst

.. raw:: html

    <h2> Availability </h2>

CHOMPACK is available as a source distribution, and precompiled binary
distributions are available for a small number of platforms (see the
:ref:`download` page).

.. raw:: html

    <h2> Authors </h2>

CHOMPACK is developed by Joachim Dahl
(:kbd:`dahl.joachim@gmail.com`), Lieven Vandenberghe
(:kbd:`vandenbe@ee.ucla.edu`), and Martin Andersen (:kbd:`martin.andersen@ucla.edu`).

