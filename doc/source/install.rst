Installation
============

.. note::

    CHOMPACK requires CVXOPT 1.2.0 or newer.


Installing a pre-built package
==============================

A pre-built binary wheel package can be installed
using `pip <https://pip.pypa.io>`_::

     pip install chompack

Wheels for Linux:

* are available for Python 2.7, 3.3, 3.4, 3.5, and 3.6 (32 and 64 bit)
* are linked against OpenBLAS

Wheels for macOS:

* are available for Python 2.7, 3.4, 3.5, and 3.6 (universal binaries)
* are linked against Accellerate BLAS/LAPACK

Wheels for Windows:

* are available for Python 2.7, 3.5, and 3.6 (64 bit only)
* are linked against MKL


Building and installing from source
===================================

The CHOMPACK Python extension can be downloaded, built, and installed by issuing the commands

.. code-block:: bash

    $ git clone https://github.com/cvxopt/chompack.git
    $ cd chompack
    $ python setup.py install 

Chompack can also be installed using pip

.. code-block:: bash

    $ pip install chompack
