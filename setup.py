try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension
from glob import glob
import os, sys
import versioneer

BLAS_NOUNDERSCORES = False
BLAS_LIB_DIR = ['/usr/lib']
BLAS_LIB = ['blas']
LAPACK_LIB = ['lapack']
BLAS_EXTRA_LINK_ARGS = []
EXTRA_COMPILE_ARGS = []
MACROS = []

BLAS_NOUNDERSCORES = int(os.environ.get("CHOMPACK_BLAS_NOUNDERSCORES",BLAS_NOUNDERSCORES)) == True
BLAS_LIB = os.environ.get("CHOMPACK_BLAS_LIB",BLAS_LIB)
LAPACK_LIB = os.environ.get("CHOMPACK_LAPACK_LIB",LAPACK_LIB)
BLAS_LIB_DIR = os.environ.get("CHOMPACK_BLAS_LIB_DIR",BLAS_LIB_DIR)
BLAS_EXTRA_LINK_ARGS = os.environ.get("CHOMPACK_BLAS_EXTRA_LINK_ARGS",BLAS_EXTRA_LINK_ARGS)
if type(BLAS_LIB) is str: BLAS_LIB = BLAS_LIB.strip().split(',')
if type(LAPACK_LIB) is str: LAPACK_LIB = LAPACK_LIB.strip().split(',')
if type(BLAS_EXTRA_LINK_ARGS) is str: BLAS_EXTRA_LINK_ARGS = BLAS_EXTRA_LINK_ARGS.strip().split(',')
if BLAS_NOUNDERSCORES: MACROS.append(('BLAS_NO_UNDERSCORE',''))

# Install Python-only reference implementation? (default: False)
py_only = os.environ.get('CHOMPACK_PY_ONLY',False)
if type(py_only) is str:
    if py_only in ['true','True','1','yes','Yes','Y','y']: py_only = True
    else: py_only = False

if os.environ.get('READTHEDOCS', False) == 'True':
    requirements = []
    py_only = True
else:
    requirements = ['cvxopt>=1.1.8']

        
# C extensions
cbase = Extension('cbase',
                  libraries = BLAS_LIB + LAPACK_LIB,
                  library_dirs = BLAS_LIB_DIR,
                  define_macros = MACROS,
                  extra_compile_args = EXTRA_COMPILE_ARGS,
                  extra_link_args = BLAS_EXTRA_LINK_ARGS,               
                  sources = glob('src/C/*.c'))

EXT_MODULES = []
if not py_only: EXT_MODULES.append(cbase)
    
setup(name='chompack',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Library for chordal matrix computations',
    author='Martin S. Andersen, Lieven Vandenberghe',
    author_email='martin.skovgaard.andersen@gmail.com, vandenbe@ee.ucla.edu',
    url='http://cvxopt.github.io/chompack/',
    download_url="https://github.com/cvxopt/chompack/archive/%s.tar.gz"%(versioneer.get_version()),
    license = 'GNU GPL version 3',
    package_dir = {"chompack": "src/python"},
    packages = ["chompack","chompack.pybase"],
    ext_package = "chompack",
    ext_modules = EXT_MODULES,
    zip_safe = False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: C',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        ],
    install_requires=requirements,
    )


