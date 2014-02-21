/* Python 2.x and 3.x interface for CHOMPACK.
 *
 * See http://wiki.python.org/moin/PortingExtensionModulesToPy3k
 * for details on combing 2.x and 3.x extension modules.
 */

#include "cvxopt.h"
#include "misc.h"

#include "chompack.h"

PyDoc_STRVAR(chompack__doc__,
    "The chompack package is a library of algorithms for sparse matrix\n"
    "problems with chordal sparsity patterns.  A symmetric sparsity pattern\n"
    "is called chordal if every positive definite matrix X with the sparsity\n"
    "pattern can be factored as X = P * L * L' * P', where P is a permutation\n"
    "matrix, and L is a lower triangular matrix with the same sparsity\n"
    "pattern as the lower triangular part of P' * X * P.  In other words,\n"
    "there exists a symmetric reordering  X[p, p] of X (with p a permutation\n"
    "of [0, 1, ..., n-1]) that has a Cholesky factorization X[p, p] = L * L'\n"
    "with zero fill-in.  Such a reordering p is called a perfect elimination\n"
    "ordering for the sparsity pattern.\n\n"
    "The following terminology and notation will be used in the documentation.\n"
    "- The projection proj(X) of a matrix X on a sparsity pattern S is the\n"
    "  sparse matrix Y with sparsity pattern S obtained by taking Y[i,j] = 0\n"
    "  if i, j is the position of a zero in S, and Y[i,j] = X[i,j] otherwise.\n"
    "- A sparsity pattern R is an embedding of the sparsity pattern S if\n"
    "  the positions of the zeros in R are a subset of the positions of the\n"
    "  zeros in S.\n"
    "- A symmetric matrix is chordal if its sparsity pattern is chordal.\n"
    "  A nonsymmetric matrix X is chordal if the sparsity pattern of X + X'\n"
    "  is chordal.\n\n"
    "The chompack package is based on two Python objects: a 'chompack'\n"
    "matrix object for storing symmetric chordal matrices, and a\n"
    "'chompack factor' object for storing Cholesky factors of positive\n"
    "definite chordal matrices.\n\n"
    "The routines in the package can be divided in different groups.\n"
    "1. Conversion from CVXOPT matrices to chompack matrices.\n"
    "   - embed(): finds a chordal embedding of the sparsity pattern of a\n"
    "     non-chordal symmetric sparse CVXOPT matrix, projects the matrix\n"
    "     on the embedding, and returns the result as a chompack matrix.\n"
    "   - project(): projects a non-chordal symmetric sparse CVXOPT matrix\n"
    "     on a given chordal sparsity pattern, and returns the result as a\n"
    "     chompack matrix.\n"
    "2. Conversion of chompack matrices to CVXOPT sparse matrices.\n"
    "   - sparse(): converts a chompack matrix or a chompack factor to\n"
    "     a CVXOPT sparse matrix.\n\n"
    "3. Main computational routines.\n"
    "   - cholesky(): Cholesky factorization of a positive definite symmetric\n"
    "     chordal matrix.\n"
    "   - completion(): maximum determinant positive definite completion of\n"
    "     of a symmetric chordal matrix.\n"
    "   - partial_inv(): projection of the inverse of a positive definite \n"
    "     chordal matrix on its sparsity pattern.\n"
    "   - hessian(): evaluates the hessian or inverse hessian of the\n"
    "     logarithmic barrier function -log det() of the cone of positive\n"
    "     semidefinite chordal matrices with a given sparsity pattern.\n\n"
    "4. Auxiliary routines for chompack matrices.\n"
    "   - copy(): makes a copy of a chompack matrix.\n"
    "   - scal(): scales a chompack matrix by a scalar.\n"
    "   - axpy(): adds a multiple of a chompack matrix to chompack matrix\n"
    "     with the same sparsity pattern.\n"
    "   - dot(): inner product of two chompack matrices with the same\n"
    "     sparsity pattern.\n"
    "   - trmv(): multiplication with a Cholesky factor.\n"
    "   - trsv(): multiplication with the inverse of a Cholesky factor.\n"
    "   - logdet(): returns 2 * log det(L) for a Cholesky factor L.\n"
    "   - info(): returns a chordal sparsity pattern as a dictionary.\n\n"
    "5. Auxiliary routines for CVXOPT sparse matrices.\n"
    "   - symmetrize(): computes X + X' - diag(diag(X)) for a lower\n"
    "     triangular sparse CVXOPT matrix X.\n"
    "   - perm(): an efficient method for computing a symmetric reordering\n"
    "     X[p, p] of a CVXOPT sparse matrix X.\n"
    "   - tril(): returns the lower triangular part of a square CVXOPT\n"
    "     sparse matrix.\n"
    "   - peo(): checks whether a given permutation is a perfect elimination\n"
    "     ordering for a CVXOPT sparse matrix.\n"
    "   - maxcardsearch(): returns the maximum cardinality search\n"
    "     reordering of a chordal CVXOPT sparse matrix X.\n\n"
    "When the documentation states the requirement that two chompack matrices\n"
    "and/or chompack factors have the same sparsity pattern, we mean by this\n"
    "that they were created from the same chompack matrix via a series of calls\n"
    "to functions that create a new chompack matrix or factor object from an\n"
    "existing one (such as cholesky(), completion(), copy(), llt(), \n"
    "partial_inv(), or project().  Pychordal matrices that were created from\n"
    "CVXOPT matrices via the functions cvxopt_to_chompack() or embed() are not\n"
    "recognized to have the same sparsity pattern, even though their sparsity\n"
    "patterns may be equal (mathematically).");

static char doc_mcs[] =
  "Maximum cardinality search ordering of a sparse chordal matrix.\n\n"
  "p = maxcardsearch(X)\n\n"
  "PURPOSE\n"
  "Returns the maximum cardinality search ordering of a symmetric\n"
  "chordal matrix X.  The maximum cardinality search ordering is a\n"
  "perfect elimination ordering in the factorization X[p, p] = L * L'.\n\n"
  "ARGUMENTS\n"
  "X         CVXOPT sparse square matrix of doubles.  Only the sparsity\n"
  "          pattern of the lower triangular part of the matrix is\n"
  "          accessed.\n\n"
  "RETURNS\n"
  "p         CVXOPT dense integer matrix of length n, if n is the order\n"
  "          of X";

static PyObject* mcs
(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *X;
  char *kwlist[] = {"X", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O:maxcardsearch", kwlist, &X))
    return NULL;

  if (!SpMatrix_Check(X) || SP_NROWS(X) != SP_NCOLS(X) || SP_ID(X) != DOUBLE )
    PY_ERR_TYPE("X must be a sparse square matrix of doubles");

  int_t j,k;
  for (j=0; j<SP_NCOLS(X); j++) {
    int has_diag = 0;
    for (k=SP_COL(X)[j]; k<SP_COL(X)[j+1]; k++)
      if (SP_ROW(X)[k] == j) { has_diag = 1; break; }

    if (!has_diag) PY_ERR_TYPE("X is missing diagonal elements");
  }

  adjgraph *A = adjgraph_create_symmetric( ((spmatrix *)X)->obj);
  if (!A) return PyErr_NoMemory();

  matrix *p = Matrix_New(SP_NROWS(X), 1, INT);
  if (!p) {
    Py_XDECREF(p);
    return PyErr_NoMemory();
  }

  int r = maxcardsearch(A, MAT_BUFI(p));
  adjgraph_destroy(A);

  if (r) return PyErr_NoMemory();

  return (PyObject *)p;
}

static char doc_peo[] =
  "Checks whether an ordering is a perfect elmimination order.\n\n"
  "peo(X, p)\n\n"
  "PURPOSE\n"
  "Returns True if the permutation p is a perfect elimination order for\n"
  "a Cholesky factorization X[p, p] = L * L'.\n\n"
  "ARGUMENTS\n"
  "X         CVXOPT sparse square matrix of doubles.  Only the sparsity\n"
  "          pattern of the lower triangular part is accessed.\n\n"
  "p         CVXOPT dense integer matrix of length n, if n is the order\n"
  "          of X";

static PyObject* peo
(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *X, *p;
  char *kwlist[] = {"X", "p", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO:peo", kwlist, &X, &p))
    return NULL;

  if (!SpMatrix_Check(X) || SP_NROWS(X) != SP_NCOLS(X) || SP_ID(X) != DOUBLE )
    PY_ERR_TYPE("X must be a sparse square matrix of doubles");

  int_t j,k;
  for (j=0; j<SP_NCOLS(X); j++) {
    int has_diag = 0;
    for (k=SP_COL(X)[j]; k<SP_COL(X)[j+1]; k++)
      if (SP_ROW(X)[k] == j) { has_diag = 1; break; }

    if (!has_diag) PY_ERR_TYPE("X is missing diagonal elements");
  }

  int_t n = SP_NCOLS(X);
  if (!Matrix_Check(p) || MAT_ID(p) != INT || MAT_NCOLS(p) != 1)
    PY_ERR_TYPE("p must be an integer nx1 matrix defining a permutation");

  if (MAT_NROWS(p) != n)
    PY_ERR_TYPE("incompatible sizes of X and p");

  int_t *ip = calloc(MAT_NROWS(p), sizeof(int_t));
  if (!ip) return PyErr_NoMemory();
  if (iperm(MAT_NROWS(p), MAT_BUFI(p), ip)) {
    free(ip);
    PY_ERR_TYPE("invalid permutation p");
  }

  adjgraph *A = adjgraph_create_symmetric(((spmatrix *)X)->obj);
  if (!A) return PyErr_NoMemory();

  int r = is_peo(A, MAT_BUFI(p), ip);
  adjgraph_destroy(A);
  free(ip);

  if (r)
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

static void free_cforest(void *F, void *descr)
{
  cliqueforest_destroy((cliqueforest *)F) ;
}

static void free_chmatrix(void *X, void *descr)
{
  PyObject *tmp = (PyObject *)((chordalmatrix *)X)->F_py;
  chordalmatrix_destroy((chordalmatrix *)X);
  Py_DECREF(tmp);
}

static char doc_ch_project[] =
  "Projects a CVXOPT sparse matrix on a chordal sparsity pattern.\n\n"
  "C = project(X, Y)\n\n"
  "PURPOSE\n"
  "Projects the CVXOPT sparse matrix Y on the sparsity pattern of\n"
  "the chompack matrix X, and returns the result as a chompack\n"
  "matrix.  Only the lower triangular part of X is referenced.\n\n"
  "ARGUMENTS\n"
  "X         chompack matrix\n\n"
  "Y         square CVXOPT sparse matrix of doubles\n\n"
  "RETURNS\n"
  "C         chompack matrix with the same sparsity pattern as X";

static PyObject* ch_project
(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *XO, *Y;
  char *kwlist[] = {"X", "Y", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO:cvxopt_to_chompack",
      kwlist, &XO, &Y))
    return NULL;

  if (!PyCObject_Check(XO))
    PY_ERR_TYPE("X is not a chompack matrix object");

  char *descr = PyCObject_GetDesc(XO);
  if (!descr || strcmp(descr, "chmatrix"))
    PY_ERR_TYPE("X is not a chompack matrix object");

  chordalmatrix *X = (chordalmatrix *) PyCObject_AsVoidPtr(XO);

  if (!(SpMatrix_Check(Y) && SP_ID(Y) == DOUBLE) &&
      !(Matrix_Check(Y) && MAT_ID(Y) == DOUBLE))
    PY_ERR_TYPE("Y must be sparse or dense cvxopt matrix of doubles");

  if (X_NROWS(Y) != X_NCOLS(Y) || X_NROWS(Y) != X->F->n)
    PY_ERR_TYPE("Incompatible dimensions");

  chordalmatrix *C;
  if (Matrix_Check(Y))
    C = dense_to_chordalmatrix(X->F, MAT_BUFD(Y));
  else {

    int k, norder = 1;
    for (k=0; k<X->F->n; k++) {
      if (X->F->p[k] != k) {
        norder = 0;
        break;
      }
    }

    if (!norder) {
      ccs *Yl = (istril(((spmatrix *)Y)->obj) ?
          ((spmatrix *)Y)->obj : tril(((spmatrix *)Y)->obj));

      if (!Yl) return PyErr_NoMemory();

      ccs *Ys = symmetrize(Yl);
      if (Yl != ((spmatrix *)Y)->obj) { free_ccs(Yl); }
      if (!Ys) return PyErr_NoMemory();

      ccs *Yp = perm(Ys, X->F->p);
      if (!Yp) {
        free_ccs(Ys);
        return PyErr_NoMemory();
      }
      free_ccs(Ys);

      C = ccs_to_chordalmatrix(X->F, Yp);
      free_ccs(Yp);
    }
    else {
      C = ccs_to_chordalmatrix(X->F, ((spmatrix *)Y)->obj);
    }
  }

  if (!C) return PyErr_NoMemory();

  C->F_py = X->F_py;
  Py_INCREF((PyObject *)C->F_py);

  return (PyObject *) PyCObject_FromVoidPtrAndDesc( (void *) C,
      "chmatrix", free_chmatrix);
}

static char doc_ch_sparse[] =
  "Converts a chompack matrix or factor to a CVXOPT sparse matrix.\n\n"
  "L = sparse(X)\n\n"
  "PURPOSE\n"
  "If X is a chompack matrix, the function returns the lower triangular\n"
  "part of X[p, p] as a CVXOPT sparse matrix, where p is permutation\n"
  "used for creating the chompack matrix with embed().  If X is chompack\n"
  "factor for a Cholesky factorization Y[p, p] = L * L', the function\n"
  "returns the lower triangular sparse matrix L.\n\n"
  "ARGUMENTS\n"
  "X         chompack matrix or factor\n\n"
  "RETURNS\n"
  "L         lower triangular CVXOPT sparse square matrix of doubles.\n";

static PyObject* ch_sparse
(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *XO = NULL;
  char *kwlist[] = {"X", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O:sparse",
      kwlist, &XO))
    return NULL;

  if (!PyCObject_Check(XO))
    PY_ERR_TYPE("X is not a chompack matrix object");

  char *descr = PyCObject_GetDesc(XO);
  if (!strcmp(descr, "chmatrix")) {

    chordalmatrix *X = (chordalmatrix *) PyCObject_AsVoidPtr(XO);

    ccs *Y = chordalmatrix_to_ccs(X);
    if (!Y) return PyErr_NoMemory();

    int i, norder = 1;
    for (i=0; i<Y->ncols; i++) {
      if (X->F->p[i] != i) {
        norder = 0;
        break;
      }
    }

    if (!norder) {
      ccs *Y2 = symmetrize(Y);
      free_ccs(Y);
      if (!Y2) return PyErr_NoMemory();

      ccs *Y3 = perm(Y2, X->F->ip);
      free_ccs(Y2);
      if (!Y3) return PyErr_NoMemory();

      Y = tril(Y3);
      free_ccs(Y3);
      if (!Y) return PyErr_NoMemory();
    }

    spmatrix *A = SpMatrix_New(0, 0, 0, DOUBLE);
    if (!A) {
      free_ccs(Y);
      return PyErr_NoMemory();
    }

    free_ccs(A->obj);
    A->obj = Y;

    return (PyObject *)A;

  } else if (!strcmp(descr, "chfactor")) {

    chordalmatrix *X = (chordalmatrix *) PyCObject_AsVoidPtr(XO);

    ccs *Y = chordalfactor_to_ccs(X);
    if (!Y) return PyErr_NoMemory();

    spmatrix *A = SpMatrix_New(0, 0, 0, DOUBLE);
    if (!A) {
      free_ccs(Y);
      return PyErr_NoMemory();
    }

    free_ccs(A->obj);
    A->obj = Y;

    return (PyObject *)A;
  }
  else PY_ERR_TYPE("X is not a chompack matrix object");
}

static char doc_ch_copy[] =
  "Returns a copy of a chompack object.\n\n"
  "Y = copy(X)\n\n"
  "ARGUMENTS\n"
  "X         chompack matrix\n\n"
  "RETURNS\n"
  "Y         chompack matrix with the same sparsity pattern as X";

static PyObject* ch_copy
(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *XO;
  char *kwlist[] = {"X", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O:copy", kwlist, &XO))
    return NULL;

  if (!PyCObject_Check(XO))
    PY_ERR_TYPE("X is not a chompack object");

  char *descr = PyCObject_GetDesc(XO);
  if (!descr || (strcmp(descr, "chmatrix") && strcmp(descr, "chfactor")))
    PY_ERR_TYPE("X is not a chompack object");

  chordalmatrix *Y, *X = (chordalmatrix *) PyCObject_AsVoidPtr(XO);

  if (!(Y = chordalmatrix_copy(X)))
    return PyErr_NoMemory();

  else {
    Py_INCREF((PyObject *)Y->F_py);
    return (PyObject *) PyCObject_FromVoidPtrAndDesc( (void *) Y,
        descr, free_chmatrix);
  }
}

static char doc_ch_cholesky[] =
  "Cholesky factorization.\n\n"
  "L = cholesky(X)\n\n"
  "PURPOSE\n"
  "Computes a zero fill-in Cholesky factorization\n\n"
  "     X[p, p] = L * L'\n\n"
  "of a positive definite chordal matrix X.\n"
  "Raises an ArithmeticError if X is not positive definite.\n\n"
  "ARGUMENTS\n"
  "X         chompack matrix\n\n"
  "RETURNS\n"
  "L         chompack factor with the same sparsity pattern as X";

static PyObject* ch_cholesky
(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *XO;
  char *kwlist[] = {"X", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O:cholesky", kwlist, &XO))
    return NULL;

  if (!PyCObject_Check(XO))
    PY_ERR_TYPE("X is not a chompack matrix object");

  char *descr = PyCObject_GetDesc(XO);
  if (!descr || strcmp(descr, "chmatrix"))
    PY_ERR_TYPE("X is not a chompack matrix object");

  chordalmatrix *X = (chordalmatrix *) PyCObject_AsVoidPtr(XO);
  chordalmatrix *Y = chordalmatrix_copy(X);
  if (!Y) return PyErr_NoMemory();

  if (cholesky(Y)) {
    chordalmatrix_destroy(Y);
    PY_ERR(PyExc_ArithmeticError, "factorization failed");
  } else {
    Py_INCREF((PyObject *)Y->F_py);
    return (PyObject *) PyCObject_FromVoidPtrAndDesc( (void *) Y,
        "chfactor", free_chmatrix);
  }
}

static char doc_ch_llt[] =
  "Computes a symmetric matrix from its Cholesky factorization.\n\n"
  "X = llt(L)\n\n"
  "PURPOSE\n"
  "Computes X from its Cholesky factorization X[p, p] = L * L'.\n\n"
  "ARGUMENTS\n"
  "L         chompack factor\n\n"
  "RETURNS\n"
  "X         chompack matrix with the same sparsity pattern as L";

static PyObject* ch_llt
(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *LO;
  char *kwlist[] = {"L", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O:llt", kwlist, &LO))
    return NULL;

  if (!PyCObject_Check(LO))
    PY_ERR_TYPE("L is not a chompack factor");

  char *descr = PyCObject_GetDesc(LO);
  if (!descr || strcmp(descr, "chfactor"))
    PY_ERR_TYPE("L is not a chompack matrix object");

  chordalmatrix *L = (chordalmatrix *) PyCObject_AsVoidPtr(LO);
  chordalmatrix *Y = chordalmatrix_copy(L);
  if (!Y) return PyErr_NoMemory();

  if (llt(Y)) {
    chordalmatrix_destroy(Y);
    return PyErr_NoMemory();
  } else {
    Py_INCREF((PyObject *)Y->F_py);
    return (PyObject *) PyCObject_FromVoidPtrAndDesc( (void *) Y,
        "chmatrix", free_chmatrix);
  }
}

static char doc_ch_partial_inv[] =
  "Evaluates the projected inverse proj(X^{-1}).\n\n"
  "Y = partial_inv(L)\n\n"
  "PURPOSE\n"
  "Computes Y = proj(X^{-1}) from the Cholesky factorization\n"
  "X[p, p] = L * L'.\n\n"
  "ARGUMENTS\n"
  "L         chompack factor\n\n"
  "RETURNS\n"
  "Y         chompack matrix with the same sparsity pattern as L\n";

static PyObject* ch_partial_inv
(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *XO;
  char *kwlist[] = {"X", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O:partial_inv", kwlist, &XO))
    return NULL;

  if (!PyCObject_Check(XO))
    PY_ERR_TYPE("X is not a chompack matrix object");

  char *descr = PyCObject_GetDesc(XO);
  if (!descr || strcmp(descr, "chfactor"))
    PY_ERR_TYPE("X is not a chompack factor object");

  chordalmatrix *X = (chordalmatrix *) PyCObject_AsVoidPtr(XO);
  chordalmatrix *Y = chordalmatrix_copy(X);
  if (!Y) return PyErr_NoMemory();

  if (partial_inv(Y)) {
    chordalmatrix_destroy(Y);
    PY_ERR(PyExc_ArithmeticError, "error computing partial inverse");
  } else {
    Py_INCREF((PyObject *)Y->F_py);
    return (PyObject *) PyCObject_FromVoidPtrAndDesc( (void *) Y,
        "chmatrix", free_chmatrix);
  }
}

static char doc_ch_completion[] =
  "Maximum-determinant positive definite completion.\n\n"
  "Y = completion(X)\n\n"
  "PURPOSE\n"
  "Returns the Cholesky factor of the inverse of the maximum-determinant\n"
  "positive definite completion of a symmetric chordal matrix X.\n\n"
  "ARGUMENTS\n"
  "X         chompack matrix\n\n"
  "RETURNS\n"
  "Y         chompack factor with the same sparsity pattern as X";

static PyObject* ch_completion
(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *XO;
  char *kwlist[] = {"X", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O:completion", kwlist, &XO))
    return NULL;

  if (!PyCObject_Check(XO))
    PY_ERR_TYPE("X is not a chompack matrix object");

  char *descr = PyCObject_GetDesc(XO);
  if (!descr || strcmp(descr, "chmatrix"))
    PY_ERR_TYPE("X is not a chompack matrix object");

  int res;
  chordalmatrix *X = (chordalmatrix *) PyCObject_AsVoidPtr(XO);

  chordalmatrix *Z = completion(X, &res);
  if (res == CHOMPACK_OK) {
    Py_INCREF((PyObject *)Z->F_py);
    return (PyObject *) PyCObject_FromVoidPtrAndDesc( (void *) Z,
        "chfactor", free_chmatrix);
  }
  else if (res == CHOMPACK_NOMEMORY)
    return PyErr_NoMemory();

  else
    PY_ERR(PyExc_ArithmeticError, "error computing completion");
}

static char doc_ch_hessian[] =
  "Evaluates the mapping H(U[i]) = proj(X^-1 * U[i] * X^-1) and its\n"
  "inverse for a list of chordal matrices U[i].\n\n"
  "hessian(L, Y, U, adj = False, inv = False)\n\n"
  "PURPOSE\n"
  "If X and U[i] are symmetric chordal matrices with the same sparsity\n"
  "pattern and with X positive definite, then the linear mapping\n\n"
  "    H(U[i]) = proj( X^{-1} * U[i] * X^{-1} )\n\n"
  "is the Hessian of the function -log(det(.)) at X applied to the\n"
  "matrix U[i].  The Hessian mapping H can be factored as\n\n"
  "    H(U[i]) = Ga( G( U[i] ))\n\n"
  "where G maps the chordal matrix U[i] to a chordal matrix with the\n"
  "same pattern, and Ga is the adjoint of G.  The purpose of the\n"
  "function hessian() is to evaluate G, Ga, and their inverses.\n"
  "The input argument L is the chompack factor of X.  The input\n"
  "argument Y is a chompack matrix containing the partial inverse\n"
  "Y = proj(X^{-1}), as returned by partial_inv().\n"
  "The input argument U is a list of chompack matrices with the same\n"
  "sparsity pattern as L and Y.\n\n"
  "On exit, U[i] is overwritten with\n\n"
  "    G(U[i])         (adj is False, inv is False)\n"
  "    G^{-1}(U[i])    (adj is False, inv is True)\n"
  "    Ga(U[i])        (adj is True, inv is False)\n"
  "    Ga^{-1}(U[i])   (adj is True, inv is True).\n\n"
  "To evaluate the Hessians H(U[i]), one can use two calls\n\n"
  "    hessian(L, Y, U, adj = False, inv = False)\n"
  "    hessian(L, Y, U, adj = True, inv = False).\n\n"
  "To evaluate the inverse Hessians H^{-1}(U[i]), one can use two calls\n\n"
  "    hessian(L, Y, U, adj = True, inv = True)\n"
  "    hessian(L, Y, U, adj = False, inv = True).\n\n"
  "ARGUMENTS\n"
  "L         chompack factor.\n\n"
  "Y         chompack matrix with the same sparsity pattern as L.\n"
  "          The partial inverse proj(X^{-1}) computed by\n"
  "          partial_inv(L).\n\n"
  "U         list of chompack matrices with the same sparsity pattern\n"
  "          as L and Y\n\n"
  "adj       True/False\n\n"
  "inv       True/False";

static PyObject* ch_hessian
(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *LO, *YO, *UO, *adj = NULL, *inv = NULL;

  char *kwlist[] = {"L", "Y", "U", "adj", "inv", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|OO:hessian",
      kwlist, &LO, &YO, &UO, &adj, &inv))
    return NULL;

  if (!PyCObject_Check(LO))
    PY_ERR_TYPE("L must be a chompack matrix");

  if (!PyCObject_Check(YO))
    PY_ERR_TYPE("Y must be a chompack matrix");

  if (!PyList_Check(UO))
    PY_ERR_TYPE("U must be a list of chompack matrices");

  if (adj && !PyBool_Check(adj))
    PY_ERR_TYPE("adj must be True or False");

  if (inv && !PyBool_Check(inv))
    PY_ERR_TYPE("inv must be True or False");

  char *descr = PyCObject_GetDesc(LO);
  if (!descr || strcmp(descr, "chfactor"))
    PY_ERR_TYPE("L must be a chompack matrix");

  descr = PyCObject_GetDesc(YO);
  if (!descr || strcmp(descr, "chmatrix"))
    PY_ERR_TYPE("Y must be a chompack matrix");

  int i, len = PyList_GET_SIZE(UO);
  chordalmatrix **U = malloc(len*sizeof(chordalmatrix *));
  if (!U) return PyErr_NoMemory();

  for (i=0; i<len; i++) {
    PyObject *Ui = PyList_GET_ITEM(UO, i);
    if (!PyCObject_Check(Ui)) {
      free(U);
      PY_ERR_TYPE("U must be a list of chompack matrices");
    }
    descr = PyCObject_GetDesc(Ui);
    if (!descr || strcmp(descr, "chmatrix")) {
      free(U);
      PY_ERR_TYPE("U must be a list of chompack matrices");
    }
    U[i] = (chordalmatrix *) PyCObject_AsVoidPtr(Ui);
  }

  chordalmatrix *L = (chordalmatrix *) PyCObject_AsVoidPtr(LO);
  chordalmatrix *Y = (chordalmatrix *) PyCObject_AsVoidPtr(YO);

  if (L->F != Y->F) {
    free(U);
    PY_ERR_TYPE("L, Y, and U must have the same chordal pattern");
  }

  for (i=0; i<len; i++) {
    if (Y->F != U[i]->F) {
      free(U);
      PY_ERR_TYPE("L, Y, and U must have the same chordal pattern");
    }
  }

  int res;
#if PY_MAJOR_VERSION >= 3
  res = hessian_factor(L, Y, U,
      adj ? PyLong_AS_LONG(adj) : 0, inv ? PyLong_AS_LONG(inv) : 0, len);

#else
  res = hessian_factor(L, Y, U,
      adj ? PyInt_AS_LONG(adj) : 0, inv ? PyInt_AS_LONG(inv) : 0, len);
#endif
  free(U);

  if (res == CHOMPACK_FACTORIZATION_ERR)
    PY_ERR(PyExc_ArithmeticError, "Hessian factorization failed");

  return Py_BuildValue("");
}

static char doc_ch_axpy[] =
  "Evaluates Y := alpha*X + Y.\n\n"
  "axpy(X, Y, alpha)\n\n"
  "ARGUMENTS\n"
  "X         chompack matrix\n\n"
  "Y         chompack matrix.  Must have the same sparsity pattern\n"
  "          as X.\n\n"
  "alpha     scalar";

static PyObject* ch_axpy
(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *XO, *YO;
  double alpha = 1.0;

  char *kwlist[] = {"X", "Y", "alpha", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|d:axpy", kwlist,
      &XO, &YO, &alpha))
    return NULL;

  if (!PyCObject_Check(XO))
    PY_ERR_TYPE("X must be a chompack matrix");

  if (!PyCObject_Check(YO))
    PY_ERR_TYPE("Y must be a chompack matrix");

  char *descr = PyCObject_GetDesc(XO);
  if (!descr || strcmp(descr, "chmatrix"))
    PY_ERR_TYPE("X must be a chompack matrix");

  descr = PyCObject_GetDesc(YO);
  if (!descr || strcmp(descr, "chmatrix"))
    PY_ERR_TYPE("Y must be a chompack matrix");

  chordalmatrix *X = (chordalmatrix *) PyCObject_AsVoidPtr(XO);
  chordalmatrix *Y = (chordalmatrix *) PyCObject_AsVoidPtr(YO);

  if (X->F != Y->F)
    PY_ERR_TYPE("X and Y must have the same chordal pattern");

  axpy(X, Y, alpha);
  return Py_BuildValue("");
}

static char doc_ch_scal[] =
  "Evaluates X := alpha * X.\n\n"
  "scal(alpha, X)\n\n"
  "ARGUMENTS\n"
  "alpha     scaling factor\n\n"
  "X         chompack matrix";

static PyObject* ch_scal
(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *XO;
  double alpha;

  char *kwlist[] = {"alpha", "X", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "dO:scal",
      kwlist, &alpha, &XO))
    return NULL;

  if (!PyCObject_Check(XO))
    PY_ERR_TYPE("X must be a chompack matrix");

  char *descr = PyCObject_GetDesc(XO);
  if (!descr || strcmp(descr, "chmatrix"))
    PY_ERR_TYPE("X must be a chompack matrix");

  chordalmatrix *X = (chordalmatrix *) PyCObject_AsVoidPtr(XO);

  scal(alpha, X);
  return Py_BuildValue("");
}

static char doc_ch_dot[] =
  "Inner product of symmetric chordal sparse matrices.\n\n"
  "z = dot(X, Y)\n\n"
  "PURPOSE\n"
  "Returns the inner product of two symmetric sparse matrices with\n"
  "the same chordal sparsity pattern.\n\n"
  "ARGUMENTS\n"
  "X         chompack matrix\n\n"
  "Y         chompack matrix.  Must have the same sparsity pattern\n"
  "          as X.\n\n"
  "RETURNS\n"
  "z         Python float";

static PyObject* ch_dot
(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *XO, *YO;

  char *kwlist[] = {"X", "Y", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO:dot", kwlist, &XO, &YO))
    return NULL;

  if (!PyCObject_Check(XO))
    PY_ERR_TYPE("X must be a chompack matrix");

  if (!PyCObject_Check(YO))
    PY_ERR_TYPE("Y must be a chompack matrix");

  char *descr = PyCObject_GetDesc(XO);
  if (!descr || strcmp(descr, "chmatrix"))
    PY_ERR_TYPE("X must be a chompack matrix");

  descr = PyCObject_GetDesc(YO);
  if (!descr || strcmp(descr, "chmatrix"))
    PY_ERR_TYPE("Y must be a chompack matrix");

  chordalmatrix *X = (chordalmatrix *) PyCObject_AsVoidPtr(XO);
  chordalmatrix *Y = (chordalmatrix *) PyCObject_AsVoidPtr(YO);

  if (X->F != Y->F)
    PY_ERR_TYPE("X and Y must have the same chordal pattern");

  return Py_BuildValue("f",dot(X,Y));
}

static char doc_ch_solve[] =
  "Solution of a triangular set of equations.\n\n"
  "solve(L, X, mode = 0)\n\n"
  "PURPOSE\n"
  "L contains the factors of a factorization P * L * L' * P' of\n"
  "a positive definite sparse chordal matrix.  On exit, X is\n"
  "overwritten with\n\n"
  "    X := L^{-1} * P   * X   (mode is 0)\n"
  "    X := P' * L^{-T}  * X   (mode is 1)\n"
  "    X := P * L  * X         (mode is 2)\n"
  "    X := L' * P' * X        (mode is 3).\n\n"
  "ARGUMENTS\n"
  "L         chompack factor\n\n"
  "X         a CVXOPT dense matrix of doubles with n rows if n is the\n"
  "          order of L\n\n"
  "mode      integer";

static PyObject* ch_solve
(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *XO, *x;
  int mode = 0;

  char *kwlist[] = {"L", "X", "mode", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|i:solve", kwlist,
      &XO, &x, &mode))
    return NULL;

  if (mode < 0 || mode > 3)
    PY_ERR_TYPE("mode must be 0, 1, 2, or 3.");

  if (!PyCObject_Check(XO))
    PY_ERR_TYPE("L must be a chompack factor");

  char *descr = PyCObject_GetDesc(XO);
  if (!descr || strcmp(descr, "chfactor"))
    PY_ERR_TYPE("L must be a chompack factor");

  chordalmatrix *X = (chordalmatrix *) PyCObject_AsVoidPtr(XO);

  if (!Matrix_Check(x) || MAT_ID(x) != DOUBLE)
    PY_ERR_TYPE("X must be a double matrix");

  if (MAT_NROWS(x) != X->F->n)
    PY_ERR_TYPE("L and X have incompatible dimensions");

  if (solve(X, MAT_BUFD(x), MAT_NCOLS(x), mode))
    return (PyObject *)PyErr_NoMemory;

  return Py_BuildValue("");
}

static char doc_ch_logdet[] =
  "Evaluates log(det(L)) for a Cholesky factor L.\n\n"
  "logdet(L)\n\n"
  "PURPOSE\n"
  "Returns the value of log(det(L)) of a Cholesky factor L.\n\n"
  "ARGUMENTS\n"
  "L         chompack factor";

static PyObject* ch_logdet
(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *LO = NULL;
  char *kwlist[] = {"L", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O:logdet", kwlist, &LO))
    return NULL;

  if (!PyCObject_Check(LO))
    PY_ERR_TYPE("L is not a chompack factor");

  char *descr = PyCObject_GetDesc(LO);
  if (!descr || strcmp(descr, "chfactor"))
    PY_ERR_TYPE("L is not a chompack factor");

  chordalmatrix *L = (chordalmatrix *) PyCObject_AsVoidPtr(LO);

  return Py_BuildValue("f", logdet(L));
}

static char doc_ch_info[] =
  "Returns information about a chordal sparsity pattern as a dictionary.\n\n"
  "info(X)\n"
  "ARGUMENTS\n"
  "X         chompack matrix or factor";

static PyObject* ch_info
(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *X;

  char *kwlist[] = {"X", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O:info", kwlist, &X))
    return NULL;

  if (!PyCObject_Check(X))
    PY_ERR_TYPE("X is not a chompack matrix or factor");

  char *descr = PyCObject_GetDesc(X);
  if (!descr || (strcmp(descr, "chmatrix") && strcmp(descr, "chfactor")))
    PY_ERR_TYPE("X is not a chompack matrix or factor");

  cliqueforest *cf = ((chordalmatrix *) PyCObject_AsVoidPtr(X))->F;

  PyObject *ret = PyList_New(cf->nCliques);
  if (!ret) return PyErr_NoMemory();

  int i, k;
  for (k=0; k<cf->nCliques; k++) {
    PyObject *So = PyList_New(nS(cf,k));
    PyObject *Uo = PyList_New(nU(cf,k));
    PyObject *Uidxo = PyList_New(nU(cf,k)); // DELETE

    PyObject *children = PyList_New(cf->list[k]->nchildren);
    if (!So || !Uo || !children) {
      //XXX
      ;
    }

    for (i=0; i<nS(cf,k); i++)
#if PY_MAJOR_VERSION >= 3
      PyList_SET_ITEM(So, i, PyLong_FromLong(S(cf,k)[i]));
#else
      PyList_SET_ITEM(So, i, PyInt_FromLong(S(cf,k)[i]));
#endif

    for (i=0; i<nU(cf,k); i++)
#if PY_MAJOR_VERSION >= 3
      PyList_SET_ITEM(Uo, i, PyLong_FromLong(U(cf,k)[i]));
#else
      PyList_SET_ITEM(Uo, i, PyInt_FromLong(U(cf,k)[i]));
#endif

    for (i=0; i<nU(cf,k); i++) // DELETE
#if PY_MAJOR_VERSION >= 3
      PyList_SET_ITEM(Uidxo, i, PyLong_FromLong(Uidx(cf,k)[i]));
#else
      PyList_SET_ITEM(Uidxo, i, PyInt_FromLong(Uidx(cf,k)[i]));
#endif

    for (i=0; i<cf->list[k]->nchildren; i++)
#if PY_MAJOR_VERSION >= 3
      PyList_SET_ITEM(children, i,
          PyLong_FromLong(cf->list[k]->children[i]->listidx));
#else
      PyList_SET_ITEM(children, i,
          PyInt_FromLong(cf->list[k]->children[i]->listidx));
#endif
    PyObject *dict = PyDict_New();
    if (!dict) {
      //XXX
      ;
    }
    PyDict_SetItemString(dict, "S", So);
    PyDict_SetItemString(dict, "U", Uo);
    PyDict_SetItemString(dict, "children", children);

    PyDict_SetItemString(dict, "Urel", Uidxo);  //DELETE

    PyList_SET_ITEM(ret, k, dict);
  }
  return ret;
}

static char doc_ch_syr1[] =
  "Computes the rank 1 update of a chordal matrix\n\n"
  "   X := alpha*proj(y*y.T) + beta*X.\n\n"
  "syr1(X, y, alpha=1.0, beta=1.0)\n\n"
  "ARGUMENTS\n"
  "X         chompack matrix\n\n"
  "y         dense nx1 matrix if n is the dimension of X\n\n"
  "alpha     float\n\n"
  "beta      float";

static PyObject* ch_syr1
(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *X, *y;
  double alpha = 1.0, beta = 1.0;
  char *kwlist[] = {"X", "y", "alpha", "beta", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|dd",
      kwlist, &X, &y, &alpha, &beta))
    return NULL;

  if (!PyCObject_Check(X))
    PY_ERR_TYPE("X is not a chompack matrix");

  char *descr = PyCObject_GetDesc(X);
  if (!descr || strcmp(descr, "chmatrix"))
    PY_ERR_TYPE("X is not a chompack matrix");

  if (!Matrix_Check(y) || MAT_ID(y) != DOUBLE)
    PY_ERR_TYPE("y must be a double matrix");

  chordalmatrix *A = (chordalmatrix *) PyCObject_AsVoidPtr(X);

  if (MAT_NROWS(y) != A->F->n || MAT_NCOLS(y) != 1)
    PY_ERR_TYPE("incompatible dimensions of y");

  syr1(A, MAT_BUFD(y), alpha, beta);

  return Py_BuildValue("");
}

static char doc_ch_syr2[] =
  "Computes the rank 2 update of a chordal matrix\n\n"
  "   X := alpha*proj(y*z.T + z*y.T) + beta*X.\n\n"
  "syr2(X, y, z, alpha=1.0, beta=1.0)\n\n"
  "ARGUMENTS\n"
  "X         chompack matrix\n\n"
  "y         dense nx1 matrix if n is the dimension of X\n\n"
  "z         dense nx1 matrix\n\n"
  "alpha     float\n\n"
  "beta      float";

static PyObject* ch_syr2
(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *X, *y, *z;
  double alpha = 1.0, beta = 1.0;
  char *kwlist[] = {"X", "y", "z", "alpha", "beta", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|dd",
      kwlist, &X, &y, &z, &alpha, &beta))
    return NULL;

  if (!PyCObject_Check(X))
    PY_ERR_TYPE("X is not a chompack matrix");

  char *descr = PyCObject_GetDesc(X);
  if (!descr || strcmp(descr, "chmatrix"))
    PY_ERR_TYPE("X is not a chompack matrix");

  if (!Matrix_Check(y) || MAT_ID(y) != DOUBLE)
    PY_ERR_TYPE("y must be a double matrix");

  if (!Matrix_Check(z) || MAT_ID(z) != DOUBLE)
    PY_ERR_TYPE("z must be a double matrix");

  chordalmatrix *A = (chordalmatrix *) PyCObject_AsVoidPtr(X);

  if (MAT_NROWS(y) != A->F->n || MAT_NCOLS(y) != 1)
    PY_ERR_TYPE("incompatible dimensions of y");

  if (MAT_NROWS(z) != A->F->n || MAT_NCOLS(z) != 1)
    PY_ERR_TYPE("incompatible dimensions of z");

  syr2(A, MAT_BUFD(y), MAT_BUFD(z), alpha, beta);

  return Py_BuildValue("");
}


static char doc_ch_perm[] =
  "Performs a symmetric permutation of a square sparse matrix.\n\n"
  "Y = perm(X, p)\n\n"
  "PURPOSE\n"
  "Equivalent to but more efficient than Y = X[p, p].\n\n"
  "ARGUMENTS\n"
  "X         CVXOPT sparse square matrix of doubles\n\n"
  "p         CVXOPT dense integer matrix of length n, if n is the order\n"
  "          of X\n\n"
  "RETURNS\n"
  "Y         CVXOPT sparse square matrix of doubles";

static PyObject* ch_perm
(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *X, *p;
  char *kwlist[] = {"X", "p", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO:perm", kwlist, &X, &p))
    return NULL;

  if (!SpMatrix_Check(X) || SP_NROWS(X) != SP_NCOLS(X) || SP_ID(X) != DOUBLE )
    PY_ERR_TYPE("X must be a sparse square matrix of doubles");

  int n = SP_NCOLS(X);
  if (!Matrix_Check(p) || MAT_ID(p) != INT || MAT_NCOLS(p) != 1)
    PY_ERR_TYPE("p must be an integer nx1 matrix defining a permutation");

  if (MAT_NROWS(p) != n)
    PY_ERR_TYPE("incompatible sizes of X and p");

  int_t *ip = calloc(MAT_NROWS(p), sizeof(int_t));
  if (!ip) return PyErr_NoMemory();
  if (iperm(MAT_NROWS(p), MAT_BUFI(p), ip)) {
    free(ip);
    PY_ERR_TYPE("invalid permutation p");
  }
  free(ip);

  spmatrix *A = SpMatrix_New(n, n, 0, DOUBLE);
  if (!A) return PyErr_NoMemory();

  ccs *obj;
  if (!(obj=perm(((spmatrix *)X)->obj, MAT_BUFI(p)))) {
    Py_DECREF(A);
    return PyErr_NoMemory();
  }

  free_ccs(((spmatrix *)A)->obj);
  ((spmatrix *)A)->obj = obj;

  return (PyObject *)A;
}

static char doc_ch_embed[] =
  "Computes a chordal embedding of a sparse matrix.\n\n"
  "Y, nfill = embed(X, p = None)\n\n"
  "PURPOSE\n"
  "Returns a chordal embedding of the sparsity pattern of X, projects\n"
  "X on the embedding, and returns the result as a chompack matrix.\n"
  "The argument p is a permutation with as default value the\n"
  "natural ordering matrix([0, 1, ..., n-1]).  The embedding is\n"
  "computed via a symbolic Cholesky factorization of X[p, p].\n\n"
  "ARGUMENTS\n"
  "X         CVXOPT sparse square matrix of doubles.  Only the lower\n"
  "          triangular part of the matrix is accessed.\n\n"
  "p         CVXOPT dense integer matrix of length n, if n is the order\n"
  "          of X\n\n"
  "RETURNS\n"
  "Y         chompack matrix.\n\n"
  "nfill     number of fill-ins in the embedding.";

static PyObject* ch_embed
(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *X, *p = NULL;
  char *kwlist[] = {"X", "p", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|O:embed", kwlist, &X, &p))
    return NULL;

  if (!SpMatrix_Check(X) || SP_NROWS(X) != SP_NCOLS(X) || SP_ID(X) != DOUBLE )
    PY_ERR_TYPE("X must be a sparse square matrix of doubles");

  ccs *A;
  int nfill;

  if (p) {

    if (!Matrix_Check(p) || MAT_ID(p) != INT || MAT_NCOLS(p) != 1)
      PY_ERR_TYPE("p must be an integer nx1 matrix defining a permutation");

    int n = SP_NROWS(X);
    if (MAT_NROWS(p) != n)
      PY_ERR_TYPE("incompatible sizes of X and p");

    int_t *ip = calloc(MAT_NROWS(p), sizeof(int_t));
    if (!ip) return PyErr_NoMemory();
    if (iperm(MAT_NROWS(p), MAT_BUFI(p), ip)) {
      free(ip);
      PY_ERR_TYPE("invalid permutation p");
    }
    free(ip);

    ccs *Xl = (istril(((spmatrix *)X)->obj) ?
        ((spmatrix *)X)->obj : tril(((spmatrix *)X)->obj));

    if (!Xl) return PyErr_NoMemory();

    ccs *Xs = symmetrize(Xl);
    if (Xl != ((spmatrix *)X)->obj) { free_ccs(Xl); }
    if (!Xs) return PyErr_NoMemory();

    ccs *Xp = perm(Xs, MAT_BUFI(p));
    if (!Xp) {
      free_ccs(Xs);
      return PyErr_NoMemory();
    }
    A = chordalembedding(Xp, &nfill);
    free_ccs(Xs);
    free_ccs(Xp);
  }
  else {

    ccs *Xl = (istril(((spmatrix *)X)->obj) ?
        ((spmatrix *)X)->obj : tril(((spmatrix *)X)->obj));

    if (!Xl) return PyErr_NoMemory();

    ccs *Xs = symmetrize(Xl);
    if (Xl != ((spmatrix *)X)->obj) { free_ccs(Xl); }
    if (!Xs) return PyErr_NoMemory();

    A = chordalembedding(Xs, &nfill);
    free_ccs(Xs);

  }
  if (!A) return PyErr_NoMemory();

  cliqueforest *F;
  if (cliqueforest_create(&F, A, p ? MAT_BUFI(p) : NULL) != CHOMPACK_OK) {
    free_ccs(A);
    return PyErr_NoMemory();
  }

  chordalmatrix *Y = ccs_to_chordalmatrix (F, A);
  if (!Y) {
    free_ccs(A);
    cliqueforest_destroy(F);
    return PyErr_NoMemory();
  }

  free_ccs(A);
  PyObject *FO = PyCObject_FromVoidPtrAndDesc( (void *) F,
      "cforest", free_cforest);

  Y->F_py = FO;

  return Py_BuildValue("Ni",PyCObject_FromVoidPtrAndDesc(Y, "chmatrix", free_chmatrix), nfill);
}

static char doc_sp_symmetrize[] =
  "Symmetrize a lower triangular matrix.\n\n"
  "Y = symmetrize(X)\n\n"
  "PURPOSE\n"
  "Returns Y := X + X.T - diag(diag(X)) where X is a lower triangular\n"
  "matrix.\n\n"
  "ARGUMENTS\n"
  "X         A CVXOPT sparse square matrix of doubles.  It must be\n"
  "          lower triangular.";

static PyObject* sp_symmetrize
(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *X;
  char *kwlist[] = {"X", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O:symmetrize", kwlist, &X))
    return NULL;

  if (!SpMatrix_Check(X) || SP_NROWS(X) != SP_NCOLS(X) || SP_ID(X) != DOUBLE )
    PY_ERR_TYPE("X must be a sparse square matrix of doubles");

  int_t j, k;
  for (j=0; j<SP_NCOLS(X); j++) {
    for (k=SP_COL(X)[j]; k<SP_COL(X)[j+1]; k++) {
      if (SP_ROW(X)[k] < j) {
        PY_ERR_TYPE("X must be lower triangular");
      } else break;
    }
  }

  ccs *A = symmetrize(((spmatrix *)X)->obj);
  if (!A) return PyErr_NoMemory();

  spmatrix *ret = SpMatrix_New(SP_NROWS(X), SP_NROWS(X), 0, DOUBLE);
  free_ccs(ret->obj);
  ret->obj = A;

  return (PyObject *)ret;
}

static char doc_sp_tril[] =
  "Returns the lower triangular part of a sparse matrix.\n\n"
  "Y = tril(X)\n\n"
  "PURPOSE\n"
  "Returns the lower triangular part of X.\n\n"
  "ARGUMENTS\n"
  "X         A CVXOPT sparse square matrix of doubles\n\n"
  "RETURNS\n"
  "Y         A CVXOPT sparse square matrix of doubles";

static PyObject* sp_tril
(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *X;
  char *kwlist[] = {"X", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O:tril", kwlist, &X))
    return NULL;

  if (!SpMatrix_Check(X) || SP_NROWS(X) != SP_NCOLS(X) || SP_ID(X) != DOUBLE )
    PY_ERR_TYPE("X must be a sparse square matrix of doubles");

  ccs *A = tril(((spmatrix *)X)->obj);
  if (!A) return PyErr_NoMemory();

  spmatrix *ret = SpMatrix_New(SP_NROWS(X), SP_NROWS(X), 0, DOUBLE);
  free_ccs(ret->obj);
  ret->obj = A;

  return (PyObject *)ret;
}

static void free_prodform_mf_factor(void *F, void *descr)
{
  free_pf_factor((pf_factor *)F);
}

static char doc_prodform_chol_symbolic[] =
  "Symbolic product-form Cholesky factorization of a sparse plus\n"
  "low-rank matrix A + B*B.T, where A is a positive semidefinite sparse\n"
  "matrix, and B is a dense matrix.\n\n"
  "F = pdf_symbolic(A, ndense, p = None, log=False)\n\n"
  "PURPOSE\n"
  "Computes the symbolic factorization of\n\n"
  "     P*L*D*L.T*P.T = A + B*B.T.\n\n"
  "A is assumed to be either lower-triangular or symmetric.\n"
  "If no permutation is given, then the AMD permutation is used.\n\n"
  "ARGUMENTS\n"
  "A         sparse semidefinite matrix of doubles\n\n"
  "ndense    number of dense columns (in B)"
  "p         permutation\n\n"
  "log       boolean switch to control output logging\n\n"
  "RETURNS\n"
  "F         An opaque object representing the symbolic factor";

static PyObject* prodform_chol_symbolic
(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *A, *p = NULL;
  int ndense = 0;
  int logoutput = 0;
  char *kwlist[] = {"A", "ndense", "p", "log", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "Oi|Oi", kwlist,
      &A, &ndense, &p, &logoutput))
    return NULL;

  if (!SpMatrix_Check(A) || SP_ID(A) != DOUBLE || SP_NROWS(A) != SP_NCOLS(A))
    PY_ERR_TYPE("A must be a square sparse matrix of doubles");

  if (ndense < 0)
    PY_ERR_TYPE("ndense must be nonnegative");

  if (p) {
    if (!Matrix_Check(p) || MAT_ID(p) != INT)
      PY_ERR_TYPE("p must be an integer matrix");

    if (MAT_NROWS(p) != SP_NROWS(A) || MAT_NCOLS(p) != 1)
      PY_ERR_TYPE("incompatible dimensions of p");

    int_t *ip = calloc(MAT_NROWS(p), sizeof(int_t));
    if (!ip) return PyErr_NoMemory();
    if (iperm(MAT_NROWS(p), MAT_BUFI(p), ip)) {
      free(ip);
      PY_ERR_TYPE("invalid permutation p");
    }
    free(ip);
  }

  ccs *Asym = (istril( ((spmatrix *)A)->obj ) ?
      symmetrize(((spmatrix *)A)->obj) : ((spmatrix *)A)->obj);
  if (!Asym) return PyErr_NoMemory();

  pf_factor *fact = pf_symbolic(Asym, p ? MAT_BUFI(p) : NULL,
      ndense, 1e-14, logoutput);

  if (Asym != ((spmatrix *)A)->obj) free_ccs(Asym);

  if (!fact) return PyErr_NoMemory();

  return (PyObject *) PyCObject_FromVoidPtrAndDesc( (void *) fact,
      "pdf_fact", free_prodform_mf_factor);
}

static char doc_prodform_chol_numeric[] =
  "Product-form Cholesky factorization of a sparse plus\n"
  "low-rank matrix\n\n"
  "      P*L*D*L.T*P.T = A + B*B.T  if transB = 'N',\n\n"
  "  or  P*L*D*L.T*P.T = A + B.T*B  if transB = 'T',\n\n"
  "where A is a positive semidefinite sparse matrix, and B is a dense\n"
  "matrix.\n\n"
  "pdf_numeric(F, A, B, transB = 'N', droptol = 1e-14)\n\n"
  "PURPOSE\n"
  "Computes the numeric factorization of A + B*B.T.  A is assumed to be\n"
  "either lower-triangular or symmetric, and elements in A outside the\n"
  "sparsity pattern used to create the symbolic factorization are\n"
  "ignored.  The symbolic factorization must have been computed by a\n"
  "call to pdf_symbolic().\n\n"
  "ARGUMENTS\n"
  "F         opaque factorization object returned by pdf_symbolic()\n\n"
  "A         sparse semidefinite matrix of doubles\n\n"
  "B         dense matrix of doubles\n\n"
  "transB    'N' or 'T'\n\n"
  "droptol   python float";

static PyObject* prodform_chol_numeric
(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *f, *A, *B;
  double droptol = 1e-14;
  char transB = 'N';
  char *kwlist[] = {"f", "A", "B", "transB", "droptol", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|cd",
      kwlist, &f, &A, &B, &transB, &droptol))
    return NULL;

  if (!PyCObject_Check(f))
    PY_ERR_TYPE("f is not an opaque product form factor");

  if (!SpMatrix_Check(A) || SP_ID(A) != DOUBLE || SP_NROWS(A) != SP_NCOLS(A))
    PY_ERR_TYPE("A must be a square sparse matrix of doubles");

  if (!Matrix_Check(B) || MAT_ID(B) != DOUBLE)
    PY_ERR_TYPE("B must be dense matrix of doubles");

  if (transB != 'N' && transB != 'T')
    PY_ERR_TYPE("ẗransB must be 'N' or 'T'");

  if ((transB == 'N' && MAT_NROWS(B) != SP_NROWS(A)) ||
      (transB == 'T' && MAT_NCOLS(B) != SP_NROWS(A)))
    PY_ERR_TYPE("incompatible dimensions of A and B");

  if (droptol < 0)
    PY_ERR(PyExc_ValueError, "droptol must be nonnegative");

  char *descr = PyCObject_GetDesc(f);
  if (!descr || strcmp(descr, "pdf_fact"))
    PY_ERR_TYPE("f is not an opaque product form factor");

  pf_factor *fact = (pf_factor *) PyCObject_AsVoidPtr(f);

  if (SP_NROWS(A) != fact->n)
    PY_ERR_TYPE("dimensions of A does not match symbolic factor");

  if ((transB == 'N' && MAT_NCOLS(B) != fact->ndense) ||
      (transB == 'T' && MAT_NROWS(B) != fact->ndense))
    PY_ERR_TYPE("dimensions of B does not match symbolic factor");

  ccs *Asym = (istril( ((spmatrix *)A)->obj ) ?
      symmetrize(((spmatrix *)A)->obj) : ((spmatrix *)A)->obj);
  if (!Asym) return PyErr_NoMemory();

  fact->droptol = droptol;

  int res = pf_numeric(fact, Asym, MAT_BUFD(B), transB);
  if (Asym != ((spmatrix *)A)->obj) free_ccs(Asym);

  if (res) PY_ERR(PyExc_ArithmeticError, "matrix is not positive definite");

  return Py_BuildValue("");
}

static char doc_prodform_chol_solve[] =
  "Solution of equations involving the product-form Cholesky factor.\n\n"
  "pdf_solve(F, b, sys = 0)\n\n"
  "If sys = 0, then the function solves P*L*D*Lj.T*P.T*xj = b.\n"
  "If sys = 1, then the function solves P*L*D^{1/2}*x = b.\n"
  "If sys = 2, then the function solves D^{1/2}*L.T*P.T*x = b.\n\n"
  "PURPOSE\n"
  "Solves a positive definite systems using product-form factorization\n"
  "computed using pdf_numeric().\n\n"
  "ARGUMENTS\n"
  "F         opaque factorization object created by pdf_numeric()\n\n"
  "b         dense matrix of doubles\n\n"
  "sys       integer";

static PyObject* prodform_chol_solve
(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *f, *b;
  int sys = 0;
  char *kwlist[] = {"f", "b", "sys", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|i", kwlist, &f, &b, &sys))
    return NULL;

  if (!PyCObject_Check(f))
    PY_ERR_TYPE("f is not an opaque product form factor");

  char *descr = PyCObject_GetDesc(f);
  if (!descr || strcmp(descr, "pdf_fact"))
    PY_ERR_TYPE("f is not an opaque product form factor");

  pf_factor *fact = (pf_factor *) PyCObject_AsVoidPtr(f);

  if (!fact->is_numeric_factor)
    PY_ERR_TYPE("f is a symbolic factor");

  if (!Matrix_Check(b) || MAT_ID(b) != DOUBLE)
    PY_ERR_TYPE("b must be a matrix of doubles");

  if (MAT_NROWS(b) != fact->n )
    PY_ERR_TYPE("incompatible dimensions of 'f' and 'b'");

  if (sys < 0 || sys > 2)
    PY_ERR_TYPE("invalid value of 'sys'");

  if (pf_solve(fact, MAT_BUFD(b), MAT_NCOLS(b), sys))
      PY_ERR(PyExc_ArithmeticError, "solve error");

  return Py_BuildValue("");
}

static char doc_prodform_chol_spsolve[] =
  "Solution of equations involving the product-form Cholesky factor.\n\n"
  "x = pdf_spsolve(F, b, sys = 0, blksize = 32)\n\n"
  "If sys = 0, then the function solves P*L*D*L.T*P.T*x = b.\n"
  "If sys = 1, then the function solves P*L*D^{1/2}*x = b.\n"
  "If sys = 2, then the function solves D^{1/2}*L.T*P.T*x = b.\n\n"
  "PURPOSE\n"
  "Solves a positive definite systems using product-form factorization\n"
  "computed using pdf_numeric().\n\n"
  "ARGUMENTS\n"
  "F         opaque factorization object created by pdf_numeric()\n\n"
  "b         dense matrix of doubles\n\n"
  "sys       integer\n\n"
  "blksize   integer specifying block column-size in righthandside";

static PyObject* prodform_chol_spsolve
(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *f, *b;
  int sys = 0, blksize=32;
  char *kwlist[] = {"f", "b", "sys", "blksize", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|ii",
      kwlist, &f, &b, &sys, &blksize))
    return NULL;

  if (!PyCObject_Check(f))
    PY_ERR_TYPE("f is not an opaque product form factor");

  char *descr = PyCObject_GetDesc(f);
  if (!descr || strcmp(descr, "pdf_fact"))
    PY_ERR_TYPE("f is not an opaque product form factor");

  pf_factor *fact = (pf_factor *) PyCObject_AsVoidPtr(f);

  if (!fact->is_numeric_factor)
    PY_ERR_TYPE("f is a symbolic factor");

  if (!SpMatrix_Check(b) || SP_ID(b) != DOUBLE)
    PY_ERR_TYPE("b must be a sparse matrix of doubles");

  if (SP_NROWS(b) != fact->n )
    PY_ERR_TYPE("incompatible dimensions of 'f' and 'b'");

  if (sys < 0 || sys > 2)
    PY_ERR_TYPE("invalid value of 'sys'");

  if (blksize < 0)
    PY_ERR_TYPE("invalid value of 'blksize'");

  PyObject *ret = (PyObject *)SpMatrix_New(SP_NROWS(b), SP_NCOLS(b), 0, DOUBLE);

  switch (pf_spsolve(fact, ((spmatrix *)b)->obj, &((spmatrix *)ret)->obj, sys, blksize)) {
  case CHOMPACK_FACTORIZATION_ERR:
    Py_DECREF(ret);
    PY_ERR(PyExc_ArithmeticError, "solve error");
  case CHOMPACK_NOMEMORY:
    Py_DECREF(ret);
    PyErr_NoMemory();
  default:
    return ret;
  }
}

static char doc_ch_maxchord[] =
  "Maximal chordal subgraph of sparsity graph.\n\n"
  "      Y = maxchord(X)\n\n"
  "  or  Y = maxchord(X, k)\n\n"
  "PURPOSE\n"
  "Computes a maximal chordal subgraph and returns the\n"
  "projection of X on the chordal subgraph. The optional\n"
  "argument k is the last vertex in a perfect elimination\n"
  "ordering associated with the chordal matrix Y. A vertex\n"
  "of maximum degree is chosen if k is omitted.\n\n"
  "ARGUMENTS\n"
  "X         sparse square matrix of doubles\n\n"
  "k         integer\n\n"
  "RETURNS\n"
  "Y         chompack matrix";

static PyObject* ch_maxchord
(PyObject *self, PyObject *args, PyObject *kwrds)
{  
  PyObject *X;
  vertex *v;
  int_t k = -1, *order, nnz;
  int i,j;

  char *kwlist[] = {"X", "k", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|n:maxchord", kwlist, &X, &k))
    return NULL;    

  if (!SpMatrix_Check(X) || SP_NROWS(X) != SP_NCOLS(X) || SP_ID(X) != DOUBLE )
    PY_ERR_TYPE("X must be a sparse square matrix of doubles");

  if (k >= ((spmatrix *)X)->obj->nrows || k < -1) 
    PY_ERR_TYPE("k must be an integer between 0 and n-1");

  for (j=0; j<SP_NCOLS(X); j++) {
    int has_diag = 0;
    for (i=SP_COL(X)[j]; i<SP_COL(X)[j+1]; i++)
      if (SP_ROW(X)[i] == j) { has_diag = 1; break; }
    if (!has_diag) PY_ERR_TYPE("X is missing diagonal elements");
  }

  ccs *Xl = (istril(((spmatrix *)X)->obj) ? ((spmatrix *)X)->obj : tril(((spmatrix *)X)->obj));
  if (!Xl) return PyErr_NoMemory();

  // compute adjgraph of X
  adjgraph *A = adjgraph_create_symmetric(Xl);
  if (!A) {
    if (Xl != ((spmatrix *)X)->obj){ free_ccs(Xl); }
    return PyErr_NoMemory();
  }
  ccs *Xs = symmetrize(Xl);
  if (Xl != ((spmatrix *)X)->obj){ free_ccs(Xl); }
  if (!Xs) {
    adjgraph_destroy(A);
    return PyErr_NoMemory();
  }

  // choose start vertex
  j = 0;
  if (k==-1) {
    k=0;
    for (i=0;i<A->numVertices;i++) {
      if (A->adjlist[i]->size > j) { k = i; j = A->adjlist[i]->size; }
    }
  }

  // find maxchord adjgraph
  adjgraph *V = maxchord(A, k, &order);
  adjgraph_destroy(A);
  if (!V) {
    free_ccs(Xs);
    return PyErr_NoMemory();
  }
  // permute Xs
  ccs *Xp = perm(Xs, order);
  free_ccs(Xs); 
  if (!Xp) {
    adjgraph_destroy(V); free(order);
    return PyErr_NoMemory();
  }

  // convert adjgraph to lower triangular ccs
  nnz = 0;
  for (i=0;i<V->numVertices;i++) nnz += V->adjlist[i]->size + 1;
  ccs *Z = alloc_ccs((int_t) V->numVertices, (int_t) V->numVertices, nnz);
  if (!Z) {
    free_ccs(Xp); adjgraph_destroy(V); free(order);
    return PyErr_NoMemory();
  }
  int_t *r = Z->rowind; int_t *c = Z->colptr; int_t t = 0;
  for (i=0;i<V->numVertices;i++) {
    r[t++] = i; // diagonal element
    v = V->adjlist[i]->head;
    for (j=0;j<V->adjlist[i]->size;j++) {
      r[t++] = v->value;
      v = v->next;
    }
    c[i+1] = t;
  }
  adjgraph_destroy(V);

  // symmetrize, permute, and create clique forrest
  ccs *Zs = symmetrize(Z);
  free_ccs(Z); 
  if (!Zs) {
    free_ccs(Xp); free(order);
    return PyErr_NoMemory();
  }
  ccs *Zp = perm(Zs, order);
  free_ccs(Zs); 
  if (!Zp) {
    free_ccs(Xp); free(order);
    return PyErr_NoMemory();
  }

  cliqueforest *F;
  if (cliqueforest_create(&F, Zp, order) != CHOMPACK_OK) {
    free_ccs(Xp); free_ccs(Zp); free(order);
    return PyErr_NoMemory();
  }
  free_ccs(Zp); free(order);

  // project Xp onto sparsity pattern
  chordalmatrix * Y = ccs_to_chordalmatrix(F,Xp);
  free_ccs(Xp);
  if (!Y) {
    cliqueforest_destroy(F);
    return PyErr_NoMemory();
  }

  // build return value
  PyObject *FO = PyCObject_FromVoidPtrAndDesc( (void *) F, "cforest", free_cforest);	
  Y->F_py = FO;

  return Py_BuildValue("N",PyCObject_FromVoidPtrAndDesc(Y, "chmatrix", free_chmatrix));
}


static PyMethodDef chompack_functions[] = {

    {"pf_symbolic", (PyCFunction)prodform_chol_symbolic,
        METH_VARARGS|METH_KEYWORDS, doc_prodform_chol_symbolic},

    {"pf_numeric", (PyCFunction)prodform_chol_numeric,
        METH_VARARGS|METH_KEYWORDS, doc_prodform_chol_numeric},

    {"pf_solve", (PyCFunction)prodform_chol_solve,
        METH_VARARGS|METH_KEYWORDS, doc_prodform_chol_solve},

    {"pf_spsolve", (PyCFunction)prodform_chol_spsolve,
        METH_VARARGS|METH_KEYWORDS, doc_prodform_chol_spsolve},

    {"syr1", (PyCFunction)ch_syr1,
        METH_VARARGS|METH_KEYWORDS, doc_ch_syr1},

    {"syr2", (PyCFunction)ch_syr2,
        METH_VARARGS|METH_KEYWORDS, doc_ch_syr2},

    {"maxcardsearch", (PyCFunction)mcs,
        METH_VARARGS|METH_KEYWORDS, doc_mcs},

    {"peo", (PyCFunction)peo,
        METH_VARARGS|METH_KEYWORDS, doc_peo},

    {"project", (PyCFunction)ch_project,
        METH_VARARGS|METH_KEYWORDS, doc_ch_project},

    {"sparse", (PyCFunction)ch_sparse,
        METH_VARARGS|METH_KEYWORDS, doc_ch_sparse},

    {"copy", (PyCFunction)ch_copy,
        METH_VARARGS|METH_KEYWORDS, doc_ch_copy},

    {"cholesky", (PyCFunction)ch_cholesky,
        METH_VARARGS|METH_KEYWORDS, doc_ch_cholesky},

    {"llt", (PyCFunction)ch_llt,
        METH_VARARGS|METH_KEYWORDS, doc_ch_llt},

    {"partial_inv", (PyCFunction)ch_partial_inv,
        METH_VARARGS|METH_KEYWORDS, doc_ch_partial_inv},

    {"completion", (PyCFunction)ch_completion,
        METH_VARARGS|METH_KEYWORDS, doc_ch_completion},

    {"hessian", (PyCFunction)ch_hessian,
        METH_VARARGS|METH_KEYWORDS, doc_ch_hessian},

    {"axpy", (PyCFunction)ch_axpy,
        METH_VARARGS|METH_KEYWORDS, doc_ch_axpy},

    {"scal", (PyCFunction)ch_scal,
        METH_VARARGS|METH_KEYWORDS, doc_ch_scal},

    {"dot", (PyCFunction)ch_dot,
        METH_VARARGS|METH_KEYWORDS, doc_ch_dot},

    {"solve", (PyCFunction)ch_solve,
        METH_VARARGS|METH_KEYWORDS, doc_ch_solve},

    {"logdet", (PyCFunction)ch_logdet,
        METH_VARARGS|METH_KEYWORDS, doc_ch_logdet},

    {"info", (PyCFunction)ch_info,
        METH_VARARGS|METH_KEYWORDS, doc_ch_info},

    {"perm", (PyCFunction)ch_perm,
        METH_VARARGS|METH_KEYWORDS, doc_ch_perm},

    {"embed", (PyCFunction)ch_embed,
        METH_VARARGS|METH_KEYWORDS, doc_ch_embed},

    {"symmetrize", (PyCFunction)sp_symmetrize,
        METH_VARARGS|METH_KEYWORDS, doc_sp_symmetrize},

    {"tril", (PyCFunction)sp_tril,
        METH_VARARGS|METH_KEYWORDS, doc_sp_tril},

    {"maxchord", (PyCFunction)ch_maxchord,
        METH_VARARGS|METH_KEYWORDS, doc_ch_maxchord},

    {NULL}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3

static PyModuleDef chompack_module = {
        PyModuleDef_HEAD_INIT,
        "chompack",
        chompack__doc__,
        -1,
        chompack_functions,
        NULL, NULL, NULL, NULL
};

#endif


PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit_chompack(void)
#else
initchompack(void)
#endif
{
  PyObject *m;

#if PY_MAJOR_VERSION >= 3
  m = PyModule_Create(&chompack_module);

  if (import_cvxopt() < 0)
    return NULL;

  return m;
#else
  m = Py_InitModule3("chompack", chompack_functions, chompack__doc__);

  if (import_cvxopt() < 0)
    return;
#endif

}
