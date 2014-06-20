/*
 * Copyright 2012 M. Andersen and L. Vandenberghe.
 * Copyright 2010-2011 L. Vandenberghe.
 * Copyright 2004-2009 J. Dahl and L. Vandenberghe.
 *
 * This file is part of CVXOPT version 1.1.5.
 *
 * CVXOPT is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * CVXOPT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "Python.h"

#ifndef __CHOMPACK__
#define __CHOMPACK__

#ifdef USE_RESTRICT
#else
#define restrict
#endif

#define int_t     Py_ssize_t

#if PY_MAJOR_VERSION >= 3
#define PYINT_CHECK(value) PyLong_Check(value)
#define PYINT_AS_LONG(value) PyLong_AS_LONG(value)
#define PYSTRING_FROMSTRING(str) PyUnicode_FromString(str)
#define PYSTRING_CHECK(a) PyUnicode_Check(a)
#define PYSTRING_COMPARE(a,b) PyUnicode_CompareWithASCIIString(a, b)
#else
#define PYINT_CHECK(value) PyInt_Check(value)
#define PYINT_AS_LONG(value) PyInt_AS_LONG(value)
#define PYSTRING_FROMSTRING(str) PyString_FromString(str)
#define PYSTRING_CHECK(a) PyString_Check(a)
#define PYSTRING_COMPARE(a,b) strcmp(PyString_AsString(a), b)
#endif


extern void dscal_(int *n, double *alpha, double *x, int *incx);
extern void dlacpy_(char *uplo, int *m, int *n, double *A, int *lda, double *B, int *ldb);
extern void dpotrf_(char *uplo, int *n, double *A, int *lda, int *info);
extern void dtrtri_(char *uplo, char *diag, int *n, double *A, int *lda, int *info);
extern void dtrtrs_(char *uplo, char *trans, char *diag, int *n, int *nrhs, double *A, int *lda, double *B, int *ldb, int *info);
extern void dtrsm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha, double *A, int *lda, double *B, int *ldb);
extern void dtrmm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha, double *A, int *lda, double *B, int *ldb);
extern void dsymm_(char *side, char *uplo, int *m, int *n, double *alpha, double *A, int *lda, double *B, int *ldb, double *beta, double *C, int *ldc);
extern void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *A, int *lda, double *B, int *ldb, double *beta, double *C, int *ldc);
extern void dsyrk_(char *uplo, char *trans, int *n, int *k, double *alpha, double *A, int *lda, double *beta, double *B, int *ldb);
extern void dsyr_(char *uplo, int *n, double *alpha, double *x, int *incx, double *A, int *lda);
extern void dlarfg_(int *n, double *alpha, double *x, int *incx, double *tau);
extern int dlarfx_(char *side, int *m, int *n, double *v, double *tau, double *C, int *ldc, double *work);

int cholesky(const int_t n,         // order of matrix
	     const int_t nsn,       // number of supernodes/cliques
	     const int_t *snpost,   // post-ordering of supernodes
	     const int_t *snptr,    // supernode pointer
	     const int_t *relptr, 
	     const int_t *relidx, 
	     const int_t *chptr, 
	     const int_t *chidx,
	     const int_t *blkptr, 
	     double * restrict blkval,
	     double * restrict fws,     // frontal matrix workspace
	     double * restrict upd,     // update matrix workspace
	     int_t * restrict upd_size  
	     );

void llt(const int_t n,         // order of matrix
	 const int_t nsn,       // number of supernodes/cliques
	 const int_t *snpost,   // post-ordering of supernodes
	 const int_t *snptr,    // supernode pointer
	 const int_t *relptr, 
	 const int_t *relidx, 
	 const int_t *chptr, 
	 const int_t *chidx,
	 const int_t *blkptr, 
	 double * restrict blkval,
	 double * restrict fws,  // frontal matrix workspace
	 double * restrict upd,  // update matrix workspace
	 int_t * restrict upd_size  
	 );


int projected_inverse(const int_t n,         // order of matrix
		      const int_t nsn,       // number of supernodes/cliques
		      const int_t *snpost,   // post-ordering of supernodes
		      const int_t *snptr,    // supernode pointer
		      const int_t *relptr, 
		      const int_t *relidx, 
		      const int_t *chptr, 
		      const int_t *chidx,
		      const int_t *blkptr, 
		      double * restrict blkval,
		      double * restrict fws,  // frontal matrix workspace
		      double * restrict upd,  // update matrix workspace
		      int_t * restrict upd_size  
		      );

int completion(const int_t n,         // order of matrix
	       const int_t nsn,       // number of supernodes/cliques
	       const int_t *snpost,   // post-ordering of supernodes
	       const int_t *snptr,    // supernode pointer
	       const int_t *relptr, 
	       const int_t *relidx, 
	       const int_t *chptr, 
	       const int_t *chidx,
	       const int_t *blkptr, 
	       double * restrict blkval,
	       double * restrict fws,  // frontal matrix workspace
	       double * restrict upd,  // update matrix workspace
	       int_t * restrict upd_size,  
	       int factored_updates);

#endif
