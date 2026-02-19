Installation
============

.. note::

    CHOMPACK requires CVXOPT 1.3.3 or newer.


Installing a pre-built package
==============================

A pre-built binary wheel package can be installed
using `pip <https://pip.pypa.io>`_::

     pip install chompack

Wheels for Linux:

* are available for Python 3.8-3.14 (64 bit)
* are linked against OpenBLAS

Wheels for macOS:

* are available for Python 3.8-3.14 (universal binaries)
* are linked against Accellerate

Wheels for Windows:

* are available for Python 3.8-3.14 (64 bit only)
* are linked against OpenBLAS


Building and installing from source
===================================

The CHOMPACK Python extension can be downloaded, built, and installed as follows:

.. code-block:: bash

    $ git clone https://github.com/cvxopt/chompack.git
    $ cd chompack
    $ pip install .
