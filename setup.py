from setuptools import setup, Extension
from glob import glob
import os, platform

BLAS_NOUNDERSCORES = 0
BLAS_LIB_DIR = ['/usr/lib']
BLAS_LIB = ['blas']
LAPACK_LIB = ['lapack']
BLAS_EXTRA_LINK_ARGS = []
EXTRA_COMPILE_ARGS = []
MACROS = []

# Guess prefix and library directories for Linux distributions
if platform.system() == "Linux":
    PREFIX = '/usr'
    # Map architecture to Debian/Ubuntu multiarch directory name
    _multiarch = {
        'x86_64':   'x86_64-linux-gnu',
        'aarch64':  'aarch64-linux-gnu',
        'i686':     'i386-linux-gnu',
        'ppc64le':  'powerpc64le-linux-gnu',
        's390x':    's390x-linux-gnu',
    }.get(platform.machine())
    if _multiarch and os.path.isdir(f"{PREFIX}/lib/{_multiarch}"):
        # Debian/Ubuntu multiarch layout
        BLAS_LIB_DIR = f"{PREFIX}/lib/{_multiarch}"
    elif os.path.isdir(f"{PREFIX}/lib64"):
        # CentOS/Fedora/RedHat/AlmaLinux layout
        BLAS_LIB_DIR = f"{PREFIX}/lib64"
    else:
        BLAS_LIB_DIR = f"{PREFIX}/lib"

BLAS_NOUNDERSCORES = int(os.environ.get("CHOMPACK_BLAS_NOUNDERSCORES", BLAS_NOUNDERSCORES)) == 1
BLAS_LIB = os.environ.get("CHOMPACK_BLAS_LIB", BLAS_LIB)
LAPACK_LIB = os.environ.get("CHOMPACK_LAPACK_LIB", LAPACK_LIB)
BLAS_LIB_DIR = os.environ.get("CHOMPACK_BLAS_LIB_DIR", BLAS_LIB_DIR)
BLAS_EXTRA_LINK_ARGS = os.environ.get("CHOMPACK_BLAS_EXTRA_LINK_ARGS", BLAS_EXTRA_LINK_ARGS)
if type(BLAS_LIB) is str:
    BLAS_LIB = BLAS_LIB.strip().split(';')
if type(BLAS_LIB_DIR) is str:
    BLAS_LIB_DIR = BLAS_LIB_DIR.strip().split(';')
if type(LAPACK_LIB) is str:
    LAPACK_LIB = LAPACK_LIB.strip().split(';')
if type(BLAS_EXTRA_LINK_ARGS) is str:
    BLAS_EXTRA_LINK_ARGS = BLAS_EXTRA_LINK_ARGS.strip().split(';')
if BLAS_NOUNDERSCORES:
    MACROS.append(('BLAS_NO_UNDERSCORE', ''))

# Install Python-only reference implementation? (default: False)
py_only = os.environ.get('CHOMPACK_PY_ONLY', False)
if type(py_only) is str:
    if py_only in ['true', 'True', '1', 'yes', 'Yes', 'Y', 'y']:
        py_only = True
    else:
        py_only = False

if os.environ.get('READTHEDOCS', False) == 'True':
    py_only = True

# C extensions
cbase = Extension('cbase',
                  libraries=LAPACK_LIB + BLAS_LIB,
                  library_dirs=BLAS_LIB_DIR,
                  define_macros=MACROS,
                  extra_compile_args=EXTRA_COMPILE_ARGS,
                  extra_link_args=BLAS_EXTRA_LINK_ARGS,
                  sources=glob('src/C/*.c'))

ext_modules = []
if not py_only:
    ext_modules.append(cbase)

setup(
    ext_modules=ext_modules,
    ext_package="chompack",
)
