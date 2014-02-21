from distutils.core import setup, Extension
import os, sys

MACROS = []
LIBRARIES = []
EXTRA_COMPILE_ARGS = []
EXT_MODULES = []

# Install Python-only reference implementation? (default: False)
py_only = os.environ.get('CHOMPACK_PY_ONLY',False)
if type(py_only) is str:
    if py_only in ['true','True','1','yes','Yes','Y','y']: py_only = True
    else: py_only = False

# C extensions
cmisc = Extension('cmisc',
                  libraries = LIBRARIES,
                  define_macros = MACROS,
                  extra_compile_args = EXTRA_COMPILE_ARGS,
                  sources = ['src/C/cmisc.c'])
if not py_only: EXT_MODULES.append(cmisc)
    
setup(name='chompack',
    version='2.0.0',
    description='Library for chordal matrix computations',
    author='Martin S. Andersen, Lieven Vandenberghe',
    author_email='martin.skovgaard.andersen@gmail.com, vandenbe@ee.ucla.edu',
    url='http://cvxopt.github.io/chompack/',
    download_url='https://github.com/cvxopt/chompack/tarball/2.0.0',
    license = 'GNU GPL version 3',
    package_dir = {"chompack": "src/python"},
    packages = ["chompack"],
    ext_package = "chompack",
    ext_modules = EXT_MODULES,
    install_requires = ['cvxopt>=1.1.6'],
    classifiers=['Development Status :: 4 - Beta',
                 'Programming Language :: Python',
                 'Programming Language :: C'])


