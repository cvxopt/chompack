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
version = '2.0.0'
from symbolic import symbolic, cspmatrix, merge_size_fill, peo
from cvxopt import spmatrix

from cholesky import cholesky
from completion import completion
from llt import llt
from projected_inverse import projected_inverse
from hessian import hessian

from misc import tril, triu, symmetrize, perm, eye
from conversion import convert_block, convert_conelp
from base import trsm, trmm, dot
from maxchord import maxchord
from mcs import maxcardsearch

__all__ = ["cspmatrix", "spmatrix", "symbolic", "peo", "cholesky", "llt", "completion", "projected_inverse", "hessian",\
           "trsm", "trmm", "tril", "triu", "version", "convert_block", "convert_conelp","maxcardsearch","maxchord","version","dot"]
