Installation
============

.. note::

   CHOMPACK requires that CVXOPT 1.1.6 or newer is installed.


Installation from source
------------------------

The CHOMPACK Python extension can be downloaded, built, and installed by issuing the commands

.. code-block:: bash

   $ git clone https://github.com/cvxopt/chompack.git
   $ cd chompack
   $ python setup.py install --user

Chompack can also be installed using pip

.. code-block:: bash
   
   $ pip install chompack


Python-only installation
-------------------------
A Python-only reference implementation
of CHOMPACK can be installed by setting the environtment variable `CHOMPACK_PY_ONLY=1`

.. code-block:: bash
   
   $ CHOMPACK_PY_ONLY=1 python install --user

or using pip,

.. code-block:: bash
   
   $ CHOMPACK_PY_ONLY=1 pip install chompack

Please note that the Python-only reference implementations are significantly slower
than the C implementations that are available in the standard installation.
