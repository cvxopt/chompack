"""
CHOMPACK --- Library for chordal matrix computations.

This file is part of Chompack.

Chompack is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Chompack is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Chompack.  If not, see <http://www.gnu.org/licenses/>.    
"""

__license__ = """This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

CVXOPT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>."""

import sys
if sys.platform == 'win32':
    import os
    extra_dll_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.libs')  
    if os.path.isdir(extra_dll_dir):  
        os.add_dll_directory(extra_dll_dir)

from chompack.symbolic import symbolic, cspmatrix, merge_size_fill, peo
from cvxopt import spmatrix

try:
    from chompack.cbase import cholesky,llt,completion,projected_inverse,hessian,trsm
    from chompack.pybase import trmm, psdcompletion, edmcompletion, mrcompletion
    __py_only__ = False
except:
    from chompack.pybase import cholesky,llt,completion,projected_inverse,hessian,trsm,trmm,psdcompletion,mrcompletion,edmcompletion
    __py_only__ = True
    
from chompack.pfcholesky import pfcholesky
from chompack.misc import tril, triu, symmetrize, perm, eye
from chompack.conversion import convert_block, convert_conelp
from chompack.base import dot, syr2
from chompack.maxchord import maxchord
from chompack.mcs import maxcardsearch

__all__ = ["cspmatrix","spmatrix","symbolic","peo","maxcardsearch","maxchord",\
           "cholesky", "llt", "completion", "psdcompletion", "edmcompletion", "mrcompletion","projected_inverse", "hessian",\
           "trsm", "trmm", "tril", "triu", "convert_block", "convert_conelp", "dot", "syr2"]

try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")