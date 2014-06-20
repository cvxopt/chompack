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
version = '2.0.1'
from chompack.symbolic import symbolic, cspmatrix, merge_size_fill, peo
from cvxopt import spmatrix

from chompack.cholesky import cholesky
from chompack.completion import completion
from chompack.llt import llt
from chompack.projected_inverse import projected_inverse
from chompack.hessian import hessian

from chompack.misc import tril, triu, symmetrize, perm, eye
from chompack.conversion import convert_block, convert_conelp
from chompack.base import trsm, trmm, dot
from chompack.maxchord import maxchord
from chompack.mcs import maxcardsearch

__all__ = ["cspmatrix", "spmatrix", "symbolic", "peo", "cholesky", "llt", "completion", "projected_inverse", "hessian",\
           "trsm", "trmm", "tril", "triu", "version", "convert_block", "convert_conelp","maxcardsearch","maxchord","version","dot"]
