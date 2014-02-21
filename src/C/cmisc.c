#include <stdlib.h>
#include "cvxopt.h"

PyDoc_STRVAR(misc__doc__, "Miscellaneous routines."); 

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



static PyMethodDef misc_functions[] = { 

  {"frontal_add_update", (PyCFunction)frontal_add_update,
   METH_VARARGS|METH_KEYWORDS, doc_frontal_add_update},

  {"frontal_get_update", (PyCFunction)frontal_get_update,
   METH_VARARGS, doc_frontal_get_update},
  
  {"lmerge", (PyCFunction)lmerge,
   METH_VARARGS, doc_lmerge},

  {NULL}  /* Sentinel */
};

PyMODINIT_FUNC
initcmisc(void)
{
  PyObject *m;

  m = Py_InitModule3("cmisc", misc_functions, misc__doc__);
  
  if (import_cvxopt() < 0)
    return;
}
