#include <stdlib.h>
#include "Python.h"
#include "chompack.h"
#include "cvxopt.h"

PyDoc_STRVAR(cbase__doc__, "Wrappers for C routines."); 

static char doc_frontal_add_update[] = 
  "frontal_add_update(F, U, relidx, relptr, i, alpha = 1.0)\n";

static PyObject* frontal_add_update
(PyObject *self, PyObject *args, PyObject *kwrds)
{

  int_t i, j, N, nf, offset;
  int idx;
  PyObject *F, *U, *relidx, *relptr;
  char *kwlist[] = {"F","U","relidx","relptr","i","alpha",NULL};
  double alpha = 1.0;

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOOOi|d", kwlist, &F, &U, &relidx, &relptr, &idx, &alpha)) return NULL;
  
  nf = MAT_NROWS(F);
  offset = MAT_BUFI(relptr)[idx];
  N = MAT_BUFI(relptr)[idx+1] - offset;
  if (!(N == MAT_NROWS(U))) return NULL;

  for (j=0; j<N; j++) {
    for (i=j; i<N; i++)
      MAT_BUFD(F)[nf*MAT_BUFI(relidx)[offset + j] + MAT_BUFI(relidx)[offset + i]] += alpha*MAT_BUFD(U)[N*j + i];
  }
 
  Py_RETURN_NONE;
}

static char doc_frontal_get_update[] = 
  "U = frontal_get_update(F, relidx, relptr, i)\n";

static PyObject* frontal_get_update
(PyObject *self, PyObject *args)
{
  
  int_t i,j,N, nf, offset;
  int idx;
  PyObject *F, *relidx, *relptr;
  matrix *U;
  
  if (!PyArg_ParseTuple(args, "OOOi", &F, &relidx, &relptr, &idx)) return NULL;
  nf = MAT_NROWS(F);
  offset = MAT_BUFI(relptr)[idx];
  N = MAT_BUFI(relptr)[idx+1] - offset;
  if (!(U = Matrix_New(N, N, DOUBLE))) return PyErr_NoMemory();

  for (j=0;j<N;j++) {
    for (i=j;i<N;i++) 
      MAT_BUFD(U)[j*N+i] = MAT_BUFD(F)[nf*MAT_BUFI(relidx)[offset + j] + MAT_BUFI(relidx)[offset + i]];
  }
  return (PyObject *) U;
}

static char doc_lmerge[] = 
  "lmerge(left, right, offsetl, offsetr , nl, nr)\n";

static PyObject* lmerge
(PyObject *self, PyObject *args)
{
  int offsetl, offsetr, nl, nr, il, ir, k, i;
  PyObject *left, *right;
  int_t *tmp;
  if (!PyArg_ParseTuple(args, "OOiiii", &left, &right, &offsetl, &offsetr, &nl, &nr)) return NULL;

  if(!(tmp = malloc(sizeof(int_t)*(nl+nr)))) return NULL;
  
  il = 0; ir = 0; k = 0;
  while (il < nl && ir < nr) {
    if (MAT_BUFI(left)[offsetl+il] < MAT_BUFI(right)[offsetr+ir]) {
      tmp[k] = MAT_BUFI(left)[offsetl+il];
      il += 1;
    }
    else if (MAT_BUFI(left)[offsetl+il] > MAT_BUFI(right)[offsetr+ir]) {
      tmp[k] = MAT_BUFI(right)[offsetr+ir];
      ir += 1;
    }
    else {
      tmp[k] = MAT_BUFI(left)[offsetl+il];
      il += 1; ir += 1; 
    }
    k += 1;
  }  
  if (il < nl) {
    for (i=0;i<nl-il;i++) tmp[k+i] = MAT_BUFI(left)[offsetl+il+i];
    k += nl-il;
  }
  if (ir < nr) {
    for (i=0;i<nr-ir;i++) tmp[k+i] = MAT_BUFI(right)[offsetr+ir+i];
    k += nr-ir;
  }
  for (i=0;i<k;i++) MAT_BUFI(left)[offsetl+i] = tmp[i];
  free(tmp);
  return Py_BuildValue("i",k);
}

static char doc_cchol[] =
  "Supernodal multifrontal Cholesky factorization:\n"
  "\n"
  ".. math::\n"
  "     X = LL^T\n"
  "\n"
  "where :math:`L` is lower-triangular. On exit, the argument :math:`X`\n"
  "contains the Cholesky factor :math:`L`.\n"
  "\n"
  ":param X:    :py:class:`cspmatrix`\n";

static PyObject* cchol
(PyObject *self, PyObject *args)
{
  int info = 0;
  int_t n, nsn, stack_depth, stack_mem, frontal_mem;
  int_t *upd_size=NULL;
  double * restrict fws=NULL, * restrict upd=NULL;
  char str_symb[] = "symb", 
    str_snpost[] = "snpost",
    str_snptr[] = "snptr",
    str_relptr[] = "relptr",
    str_relidx[] = "relidx",
    str_chptr[] = "chptr",
    str_chidx[] = "chidx",
    str_blkptr[] = "blkptr",
    str_blkval[] = "blkval",
    str_memory[] = "memory",
    str_stack_depth[] = "stack_depth",
    str_stack_mem[] = "stack_mem",
    str_frontal_mem[] = "frontal_mem",
    str_is_factor[] = "is_factor",
    str_n[] = "n",
    str_nsn[] = "Nsn";

  PyObject *A, *symb, *Py_snpost, *Py_snptr, *Py_relptr, *Py_relidx,
    *Py_chptr, *Py_chidx, *Py_blkptr, *Py_blkval, *PyObj, *Py_memory;

  // extract pointers from cspmatrix A
  if (!PyArg_ParseTuple(args, "O", &A)) return NULL;  // A : borrowed reference
  Py_blkval = PyObject_GetAttrString(A, str_blkval);

  // extract pointers and values from symbolic object
  symb = PyObject_GetAttrString(A,str_symb); 
  Py_snpost = PyObject_GetAttrString(symb, str_snpost);
  Py_snptr  = PyObject_GetAttrString(symb, str_snptr);
  Py_relptr = PyObject_GetAttrString(symb, str_relptr);
  Py_relidx = PyObject_GetAttrString(symb, str_relidx);
  Py_chptr  = PyObject_GetAttrString(symb, str_chptr);
  Py_chidx  = PyObject_GetAttrString(symb, str_chidx);
  Py_blkptr = PyObject_GetAttrString(symb, str_blkptr);
  PyObj = PyObject_GetAttrString(symb, str_n);
  n   = PYINT_AS_LONG(PyObj); Py_DECREF(PyObj);
  PyObj = PyObject_GetAttrString(symb, str_nsn);
  nsn = PYINT_AS_LONG(PyObj); Py_DECREF(PyObj);
  Py_memory = PyObject_GetAttrString(symb, str_memory);
  stack_depth = PYINT_AS_LONG(PyDict_GetItemString(Py_memory, str_stack_depth));
  stack_mem   = PYINT_AS_LONG(PyDict_GetItemString(Py_memory, str_stack_mem));
  frontal_mem = PYINT_AS_LONG(PyDict_GetItemString(Py_memory, str_frontal_mem));
  Py_DECREF(Py_memory);
  Py_DECREF(symb);

  // check that cspmatrix factor flag is False
  PyObj = PyObject_GetAttrString(A,str_is_factor);
  if (PyObj == Py_False) {
    Py_DECREF(PyObj);
  } 
  else {
    Py_DECREF(PyObj);
    return PyErr_Format(PyExc_ValueError,"X must be a cspmatrix");
  }

  // allocate workspace
  if (!(upd = malloc(stack_mem*sizeof(double)))) return PyErr_NoMemory();
  if (!(fws = malloc(frontal_mem*sizeof(double)))) {
    free(upd);
    return PyErr_NoMemory();
  }  
  if (!(upd_size = malloc(stack_depth*sizeof(int_t)))) {
    free(upd);
    free(fws);
    return PyErr_NoMemory();
  }
  
  // call numerical cholesky
  info = cholesky(n,nsn,MAT_BUFI(Py_snpost),MAT_BUFI(Py_snptr),
		  MAT_BUFI(Py_relptr),MAT_BUFI(Py_relidx),
		  MAT_BUFI(Py_chptr),MAT_BUFI(Py_chidx),
		  MAT_BUFI(Py_blkptr),MAT_BUFD(Py_blkval),
		  fws,upd,upd_size);

  // update reference counts
  Py_DECREF(Py_snpost); Py_DECREF(Py_snptr); 
  Py_DECREF(Py_relptr); Py_DECREF(Py_relidx); 
  Py_DECREF(Py_chptr); Py_DECREF(Py_chidx);
  Py_DECREF(Py_blkptr); Py_DECREF(Py_blkval);

  // free workspace
  free(fws); free(upd); free(upd_size);

  // set cspmatrix factor flag to True
  PyObject_SetAttrString(A, str_is_factor, Py_True);

  // check for errors
  if (info) return PyErr_Format(PyExc_ValueError,"factorization failed");

  return Py_BuildValue("");
}


static char doc_cllt[] =
  "Supernodal multifrontal Cholesky product:\n"
  "\n"
  ".. math::\n"
  "     X = LL^T\n"
  "\n"
  "where :math:`L` is lower-triangular. On exit, the argument `L`\n"
  "contains the product :math:`X`.\n"
  "\n"
  ":param L:    :py:class:`cspmatrix` (factor)";

static PyObject* cllt
(PyObject *self, PyObject *args)
{
  int_t n, nsn, stack_depth, stack_mem, frontal_mem;
  int_t *upd_size=NULL;
  double * restrict fws=NULL, * restrict upd=NULL;
  char str_symb[] = "symb", 
    str_snpost[] = "snpost",
    str_snptr[] = "snptr",
    str_relptr[] = "relptr",
    str_relidx[] = "relidx",
    str_chptr[] = "chptr",
    str_chidx[] = "chidx",
    str_blkptr[] = "blkptr",
    str_blkval[] = "blkval",
    str_memory[] = "memory",
    str_stack_depth[] = "stack_depth",
    str_stack_mem[] = "stack_mem",
    str_frontal_mem[] = "frontal_mem",
    str_is_factor[] = "is_factor",
    str_n[] = "n",
    str_nsn[] = "Nsn";

  PyObject *A, *symb, *Py_snpost, *Py_snptr, *Py_relptr, *Py_relidx,
    *Py_chptr, *Py_chidx, *Py_blkptr, *Py_blkval, *PyObj, *Py_memory;

  // extract pointers from cspmatrix A
  if (!PyArg_ParseTuple(args, "O", &A)) return NULL;  // A : borrowed reference
  Py_blkval = PyObject_GetAttrString(A, str_blkval);

  // extract pointers and values from symbolic object
  symb = PyObject_GetAttrString(A,str_symb); 
  Py_snpost = PyObject_GetAttrString(symb, str_snpost);
  Py_snptr  = PyObject_GetAttrString(symb, str_snptr);
  Py_relptr = PyObject_GetAttrString(symb, str_relptr);
  Py_relidx = PyObject_GetAttrString(symb, str_relidx);
  Py_chptr  = PyObject_GetAttrString(symb, str_chptr);
  Py_chidx  = PyObject_GetAttrString(symb, str_chidx);
  Py_blkptr = PyObject_GetAttrString(symb, str_blkptr);
  PyObj = PyObject_GetAttrString(symb, str_n);
  n   = PYINT_AS_LONG(PyObj); Py_DECREF(PyObj);
  PyObj = PyObject_GetAttrString(symb, str_nsn);
  nsn = PYINT_AS_LONG(PyObj); Py_DECREF(PyObj);
  Py_memory = PyObject_GetAttrString(symb, str_memory);
  stack_depth = PYINT_AS_LONG(PyDict_GetItemString(Py_memory, str_stack_depth));
  stack_mem   = PYINT_AS_LONG(PyDict_GetItemString(Py_memory, str_stack_mem));
  frontal_mem = PYINT_AS_LONG(PyDict_GetItemString(Py_memory, str_frontal_mem));
  Py_DECREF(Py_memory);
  Py_DECREF(symb);

  // check that cspmatrix factor flag is True
  PyObj = PyObject_GetAttrString(A,str_is_factor);
  if (PyObj == Py_True) {
    Py_DECREF(PyObj);
  } 
  else {
    Py_DECREF(PyObj);
    return PyErr_Format(PyExc_ValueError,"X must be a cspmatrix");
  }

  // allocate workspace
  if (!(upd = malloc(stack_mem*sizeof(double)))) return PyErr_NoMemory();
  if (!(fws = malloc(frontal_mem*sizeof(double)))) {
    free(upd);
    return PyErr_NoMemory();
  }  
  if (!(upd_size = malloc(stack_depth*sizeof(int_t)))) {
    free(upd);
    free(fws);
    return PyErr_NoMemory();
  }
  
  // call llt
  llt(n,nsn,MAT_BUFI(Py_snpost),MAT_BUFI(Py_snptr),
      MAT_BUFI(Py_relptr),MAT_BUFI(Py_relidx),
      MAT_BUFI(Py_chptr),MAT_BUFI(Py_chidx),
      MAT_BUFI(Py_blkptr),MAT_BUFD(Py_blkval),
      fws,upd,upd_size);

  // update reference counts
  Py_DECREF(Py_snpost); Py_DECREF(Py_snptr); 
  Py_DECREF(Py_relptr); Py_DECREF(Py_relidx); 
  Py_DECREF(Py_chptr); Py_DECREF(Py_chidx);
  Py_DECREF(Py_blkptr); Py_DECREF(Py_blkval);

  // free workspace
  free(fws); free(upd); free(upd_size);

  // set cspmatrix factor flag to False
  PyObject_SetAttrString(A, str_is_factor, Py_False);

  return Py_BuildValue("");
}


static char doc_cprojected_inverse[] =
  "Supernodal multifrontal projected inverse. The routine computes the projected inverse\n"
  "\n"
  ".. math::\n"
  "     Y = P(L^{-T}L^{-1}) \n"
  "\n"
  "where :math:`L` is a Cholesky factor. On exit, the argument :math:`L` contains the\n"
  "projected inverse :math:`Y`.\n"
  "\n"
  ":param L:            :py:class:`cspmatrix` (factor)";

static PyObject* cprojected_inverse
(PyObject *self, PyObject *args)
{
  int info = 0;
  int_t n, nsn, stack_depth, stack_mem, frontal_mem;
  int_t *upd_size=NULL;
  double * restrict fws=NULL, * restrict upd=NULL;
  char str_symb[] = "symb", 
    str_snpost[] = "snpost",
    str_snptr[] = "snptr",
    str_relptr[] = "relptr",
    str_relidx[] = "relidx",
    str_chptr[] = "chptr",
    str_chidx[] = "chidx",
    str_blkptr[] = "blkptr",
    str_blkval[] = "blkval",
    str_memory[] = "memory",
    str_stack_depth[] = "stack_depth",
    str_stack_mem[] = "stack_mem",
    str_frontal_mem[] = "frontal_mem",
    str_is_factor[] = "is_factor",
    str_n[] = "n",
    str_nsn[] = "Nsn";

  PyObject *A, *symb, *Py_snpost, *Py_snptr, *Py_relptr, *Py_relidx,
    *Py_chptr, *Py_chidx, *Py_blkptr, *Py_blkval, *PyObj, *Py_memory;

  // extract pointers from cspmatrix A
  if (!PyArg_ParseTuple(args, "O", &A)) return NULL;  // A : borrowed reference
  Py_blkval = PyObject_GetAttrString(A, str_blkval);

  // extract pointers and values from symbolic object
  symb = PyObject_GetAttrString(A,str_symb); 
  Py_snpost = PyObject_GetAttrString(symb, str_snpost);
  Py_snptr  = PyObject_GetAttrString(symb, str_snptr);
  Py_relptr = PyObject_GetAttrString(symb, str_relptr);
  Py_relidx = PyObject_GetAttrString(symb, str_relidx);
  Py_chptr  = PyObject_GetAttrString(symb, str_chptr);
  Py_chidx  = PyObject_GetAttrString(symb, str_chidx);
  Py_blkptr = PyObject_GetAttrString(symb, str_blkptr);
  PyObj = PyObject_GetAttrString(symb, str_n);
  n   = PYINT_AS_LONG(PyObj); Py_DECREF(PyObj);
  PyObj = PyObject_GetAttrString(symb, str_nsn);
  nsn = PYINT_AS_LONG(PyObj); Py_DECREF(PyObj);
  Py_memory = PyObject_GetAttrString(symb, str_memory);
  stack_depth = PYINT_AS_LONG(PyDict_GetItemString(Py_memory, str_stack_depth));
  stack_mem   = PYINT_AS_LONG(PyDict_GetItemString(Py_memory, str_stack_mem));
  frontal_mem = PYINT_AS_LONG(PyDict_GetItemString(Py_memory, str_frontal_mem));
  Py_DECREF(Py_memory);
  Py_DECREF(symb);

  // check that cspmatrix factor flag is True
  PyObj = PyObject_GetAttrString(A,str_is_factor);
  if (PyObj == Py_True) {
    Py_DECREF(PyObj);
  } 
  else {
    Py_DECREF(PyObj);
    return PyErr_Format(PyExc_ValueError,"X must be a cspmatrix");
  }

  // allocate workspace
  if (!(upd = malloc(stack_mem*sizeof(double)))) return PyErr_NoMemory();
  if (!(fws = malloc(frontal_mem*sizeof(double)))) {
    free(upd);
    return PyErr_NoMemory();
  }  
  if (!(upd_size = malloc(stack_depth*sizeof(int_t)))) {
    free(upd);
    free(fws);
    return PyErr_NoMemory();
  }
  
  // call projected_inverse
  info = projected_inverse(n,nsn,MAT_BUFI(Py_snpost),MAT_BUFI(Py_snptr),
			   MAT_BUFI(Py_relptr),MAT_BUFI(Py_relidx),
			   MAT_BUFI(Py_chptr),MAT_BUFI(Py_chidx),
			   MAT_BUFI(Py_blkptr),MAT_BUFD(Py_blkval),
			   fws,upd,upd_size);
  
  // update reference counts
  Py_DECREF(Py_snpost); Py_DECREF(Py_snptr); 
  Py_DECREF(Py_relptr); Py_DECREF(Py_relidx); 
  Py_DECREF(Py_chptr); Py_DECREF(Py_chidx);
  Py_DECREF(Py_blkptr); Py_DECREF(Py_blkval);

  // free workspace
  free(fws); free(upd); free(upd_size);

  // set cspmatrix factor flag to False
  PyObject_SetAttrString(A, str_is_factor, Py_False);

  // check for errors
  if (info) return PyErr_Format(PyExc_ValueError,"partial inverse failed");

  return Py_BuildValue("");
}

static char doc_ccompletion[] =
  "Supernodal multifrontal maximum determinant positive definite\n"
  "matrix completion. The routine computes the Cholesky factor\n"
  ":math:`L` of the inverse of the maximum determinant positive\n"
  "definite matrix completion of :math:`X`:, i.e.,\n"
  "\n"
  ".. math::\n"
  "     P( S^{-1} ) = X\n"
  "\n"
  "where :math:`S = LL^T`. On exit, the argument `X` contains the\n"
  "lower-triangular Cholesky factor :math:`L`.\n"
  "\n"
  "The optional argument `factored_updates` can be used to enable (if\n"
  "True) or disable (if False) updating of intermediate\n"
  "factorizations.\n"
  "\n"
  ":param X:                 :py:class:`cspmatrix`\n"
  ":param factored_updates:  boolean";

static PyObject* ccompletion
(PyObject *self, PyObject *args, PyObject *kwrds)
{
  int info = 0, factored_updates = 0;
  int_t n, nsn, stack_depth, stack_mem, frontal_mem;
  int_t *upd_size=NULL;
  double * restrict fws=NULL, * restrict upd=NULL;
  char str_symb[] = "symb", 
    str_snpost[] = "snpost",
    str_snptr[] = "snptr",
    str_relptr[] = "relptr",
    str_relidx[] = "relidx",
    str_chptr[] = "chptr",
    str_chidx[] = "chidx",
    str_blkptr[] = "blkptr",
    str_blkval[] = "blkval",
    str_memory[] = "memory",
    str_stack_depth[] = "stack_depth",
    str_stack_mem[] = "stack_mem",
    str_frontal_mem[] = "frontal_mem",
    str_is_factor[] = "is_factor",
    str_n[] = "n",
    str_nsn[] = "Nsn";
  char *kwlist[] = {"X","factored_updates",NULL};

  PyObject *A, *symb, *Py_snpost, *Py_snptr, *Py_relptr, *Py_relidx,
    *Py_chptr, *Py_chidx, *Py_blkptr, *Py_blkval, *Py_memory, *PyObj;
  PyObj = Py_True;

  // extract pointers from arguments
  if(!PyArg_ParseTupleAndKeywords(args, kwrds, "O|O", kwlist, &A, &PyObj)) return NULL;
  if (PyObj == Py_True) factored_updates = 1;
  else factored_updates = 0;
  Py_blkval = PyObject_GetAttrString(A, str_blkval);

  // extract pointers and values from symbolic object
  symb = PyObject_GetAttrString(A,str_symb); 
  Py_snpost = PyObject_GetAttrString(symb, str_snpost);
  Py_snptr  = PyObject_GetAttrString(symb, str_snptr);
  Py_relptr = PyObject_GetAttrString(symb, str_relptr);
  Py_relidx = PyObject_GetAttrString(symb, str_relidx);
  Py_chptr  = PyObject_GetAttrString(symb, str_chptr);
  Py_chidx  = PyObject_GetAttrString(symb, str_chidx);
  Py_blkptr = PyObject_GetAttrString(symb, str_blkptr);
  PyObj = PyObject_GetAttrString(symb, str_n);
  n   = PYINT_AS_LONG(PyObj); Py_DECREF(PyObj);
  PyObj = PyObject_GetAttrString(symb, str_nsn);
  nsn = PYINT_AS_LONG(PyObj); Py_DECREF(PyObj);
  Py_memory = PyObject_GetAttrString(symb, str_memory);
  stack_depth = PYINT_AS_LONG(PyDict_GetItemString(Py_memory, str_stack_depth));
  stack_mem   = PYINT_AS_LONG(PyDict_GetItemString(Py_memory, str_stack_mem));
  frontal_mem = PYINT_AS_LONG(PyDict_GetItemString(Py_memory, str_frontal_mem));
  Py_DECREF(Py_memory);
  Py_DECREF(symb);

  // check that cspmatrix factor flag is True
  PyObj = PyObject_GetAttrString(A,str_is_factor);
  if (PyObj == Py_False) {
    Py_DECREF(PyObj);
  } 
  else {
    Py_DECREF(PyObj);
    return PyErr_Format(PyExc_ValueError,"X must be a cspmatrix");
  }

  // allocate workspace
  if (!(upd = malloc(stack_mem*sizeof(double)))) return PyErr_NoMemory();
  if (!(fws = malloc(frontal_mem*sizeof(double)))) {
    free(upd);
    return PyErr_NoMemory();
  }  
  if (!(upd_size = malloc(stack_depth*sizeof(int_t)))) {
    free(upd);
    free(fws);
    return PyErr_NoMemory();
  }
  
  // call completion
  info = completion(n,nsn,MAT_BUFI(Py_snpost),MAT_BUFI(Py_snptr),
		    MAT_BUFI(Py_relptr),MAT_BUFI(Py_relidx),
		    MAT_BUFI(Py_chptr),MAT_BUFI(Py_chidx),
		    MAT_BUFI(Py_blkptr),MAT_BUFD(Py_blkval),
		    fws,upd,upd_size,factored_updates);
  
  // update reference counts
  Py_DECREF(Py_snpost); Py_DECREF(Py_snptr); 
  Py_DECREF(Py_relptr); Py_DECREF(Py_relidx); 
  Py_DECREF(Py_chptr); Py_DECREF(Py_chidx);
  Py_DECREF(Py_blkptr); Py_DECREF(Py_blkval);

  // free workspace
  free(fws); free(upd); free(upd_size);

  // set cspmatrix factor flag to False
  PyObject_SetAttrString(A, str_is_factor, Py_True);

  // check for errors
  if (info) return PyErr_Format(PyExc_ValueError,"completion failed");

  return Py_BuildValue("");
}

static PyMethodDef cbase_functions[] = { 

  {"frontal_add_update", (PyCFunction)frontal_add_update,
   METH_VARARGS|METH_KEYWORDS, doc_frontal_add_update},

  {"frontal_get_update", (PyCFunction)frontal_get_update,
   METH_VARARGS, doc_frontal_get_update},
  
  {"lmerge", (PyCFunction)lmerge,
   METH_VARARGS, doc_lmerge},

  {"cholesky", (PyCFunction)cchol,
   METH_VARARGS, doc_cchol},

  {"llt", (PyCFunction)cllt,
   METH_VARARGS, doc_cllt},

  {"projected_inverse", (PyCFunction)cprojected_inverse,
   METH_VARARGS, doc_cprojected_inverse},

  {"completion", (PyCFunction)ccompletion,
   METH_VARARGS|METH_KEYWORDS, doc_ccompletion},

  {NULL}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
// Python 3.x
static PyModuleDef cbase_module = {
    PyModuleDef_HEAD_INIT,
    "cbase",
    cbase__doc__,
    -1,
    cbase_functions,
    NULL, NULL, NULL, NULL
};
PyMODINIT_FUNC PyInit_cbase(void) {
  PyObject *cbase_mod;
  cbase_mod = PyModule_Create(&cbase_module);
  if (cbase_mod == NULL)
    return NULL;
  if (import_cvxopt() < 0) 
    return NULL;
  return cbase_mod;
}
#else
// Python 2.x
PyMODINIT_FUNC initcbase(void) {
  PyObject *m;
  m = Py_InitModule3("cbase", cbase_functions, cbase__doc__); 
  if (import_cvxopt() < 0) 
    return;
}
#endif


 
