from distutils.core import setup, Extension
import sys, glob

ATLAS_LIB_DIR = '/usr/lib'

MACROS = []
CVXOPT_SRC = 'cvxopt-1.1.6/src/C'    
if sys.maxsize > 2**31: MACROS += [('DLONG','')]

LIBRARIES = ['lapack','blas','gomp']
EXTRA_COMPILE_ARGS = ['-fopenmp']

AMD_SRC = glob.glob(CVXOPT_SRC + '/SuiteSparse/AMD/Source/*.c')
AMD_DIR = [CVXOPT_SRC + '/SuiteSparse/AMD/Include/',
           CVXOPT_SRC + '/SuiteSparse/SuiteSparse_config/']
 
# Compile without OpenMP under Mac OS
if sys.platform.startswith('darwin'):
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
                                 'maxchord.c'] + AMD_SRC
                    )

setup (name = 'chompack', 
    version = '1.1.1',
    description = 'Library for chordal matrix computations',
    ext_modules = [chompack],
    author = 'Joachim Dahl, Lieven Vandenberghe, Martin Andersen',
    author_email = 'dahl.joachim@gmail.com, vandenbe@ee.ucla.edu, martin.skovgaard.andersen@gmail.com',
    url = 'http://abel.ee.ucla.edu/chompack',
    install_requires = ['cvxopt>=1.1.6'],
    classifiers=['Development Status :: 5 - Production/Stable',
                 'Programming Language :: Python',
                 'Programming Language :: C'])
