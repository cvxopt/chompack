.. c_installation:

************
Installation 
************

Python interface installation
=============================

The Python interface requires that Python 2.5+ is installed, as well as the
source code of CVXOPT 1.1.3.

To install the interface, the `CVXOPT_SRC` and `ATLAS_LIB_DIR` variables
in lines 4 and 5 of the `setup.py` script must be updated to point on the 
`CVXOPT` source and `BLAS/LAPACK` (or `ATLAS`) library locations, respectively.

.. code-block:: python
    :linenos:

    from distutils.core import setup, Extension
    import sys, glob

    ATLAS_LIB_DIR = '/usr/lib'

    MACROS = []
    if sys.version_info[0] == 2:
        CVXOPT_SRC = '../cvxopt/src/C'
    else: 
        CVXOPT_SRC = '../cvxopt3/src/C'
    
    if sys.maxsize > 2**31: MACROS += [('DLONG','')]

    LIBRARIES = ['lapack','blas','gomp']
    EXTRA_COMPILE_ARGS = ['-fopenmp']

    AMD_SRC = glob.glob(CVXOPT_SRC + '/SuiteSparse/AMD/Source/*.c')
    AMD_DIR = [CVXOPT_SRC + '/SuiteSparse/AMD/Include/',
               CVXOPT_SRC + '/SuiteSparse/UFconfig/']
 
    # Compile without OpenMP under Mac OS 10.5.x (default: GCC 4.0.1, Python 2.5)
    if sys.platform.startswith('darwin') and not sys.version >= '2.6':
        LIBRARIES.remove('gomp')
        EXTRA_COMPILE_ARGS.remove('-fopenmp')

    EXTRA_COMPILE_ARGS.append('-msse3')
    EXTRA_COMPILE_ARGS.append('-O3')

    chompack = Extension('chompack',
                      include_dirs = [ CVXOPT_SRC ] + AMD_DIR,
                      libraries = LIBRARIES,
                      library_dirs = [ ATLAS_LIB_DIR, '.' ],
                      define_macros = MACROS,
                      extra_compile_args = EXTRA_COMPILE_ARGS,
                      sources = ['pychompack.c',
                                 'chompack.c',
                                 'maxcardsearch.c',
                                 'cliquetree.c',
                                 'sparse.c',
                                 'adjgraph.c',
                                 'cholupdate.c',
                                 'pf_cholesky.c',
                                 'maxchord.c'] + AMD_SRC)

    setup (name = 'chompack', 
        version = '1.1.1', 
        ext_modules = [chompack])

On Linux, the Python package is built and installed by issuing the command

::
  
    $ python setup.py install --user 
    
or it can be installed for all users by writing

::

    $ sudo python setup.py install


C library installation
======================

The C library is developed and tested using Mac OS X Lion and Ubuntu 11.04 for
both 32 and 64 bit. To compile with OpenMP, GCC 4.3 or newer should be used. 

The C library is built using `SCons <http://www.scons.org>`_,
with must be installed separately. 
The library requires that the `BLAS <http://www.netlib.org/blas>`_ and 
`LAPACK <http://www.netlib.org/lapack>`_  
libraries are already installed in a directory known by the C linker; 
otherwise line 8 of the `SConstruct` build script

.. code-block:: python
    :linenos:
    
    env = Environment()

    SRCS = 'adjgraph.c chompack.c cliquetree.c maxcardsearch.c sparse.c cholupdate.c pf_cholesky.c'

    import sys
    if sys.maxint > 2**31: env.Append(CPPDEFINES = ['DLONG'])
	
    #env.Append(LIBPATH = ['/usr/lib'])

    omp = ARGUMENTS.get('OMP', 0)
    if int(omp):
        env.Append(CCFLAGS = '-fopenmp', LIBS = ['gomp'])

    chompack = env.SharedLibrary( 'chompack', Split(SRCS))
    env.Program('example', 'example.c', LIBS = [chompack, 'blas', 'lapack'])  
    
must be updated to reflect the installation directory for those libraries.
By default the library is built without 
`OpenMP <http://www.openmp.org>`_ multiprocessor by issuing the
`SCons` command

.. code-block:: bash 

    scons 

Multiprocessor support is included by issuing the command

.. code-block:: bash 

    scons OMP=1

and requires that the `openmp` library is installed in a location known 
to the C linker; otherwise the library location must be updated in 
line 8 of the `SConstruct` build script. Compilation on other platforms 
than Ubuntu Linux should be straighforward using SCons,  but the OpenMP 
configuration might need to be modified.

