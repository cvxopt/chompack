/* Miscellaneous routines for sparse matrices taken from CVXOPT */

#include <stdio.h>
#include <stdlib.h>

#include "chompack.h"
#include "sparse.h"
#include "chtypes.h"

ccs * alloc_ccs(int_t nrows, int_t ncols, int nnz)
{
  ccs *obj = malloc(sizeof(ccs));
  if (!obj) return NULL;

  obj->nrows = nrows;
  obj->ncols = ncols;
  obj->id = DOUBLE;

  obj->values = malloc(sizeof(double)*nnz);
  obj->colptr = calloc(ncols+1,sizeof(int_t));
  obj->rowind = malloc(sizeof(int_t)*nnz);

  if (!obj->values || !obj->colptr || !obj->rowind) {
    free(obj->values); free(obj->colptr); free(obj->rowind); free(obj);
    return NULL;
  }

  return obj;
}

void free_ccs(ccs *obj) {
  free(obj->values);
  free(obj->rowind);
  free(obj->colptr);
  free(obj);
}


ccs * perm(const ccs *A, int_t *p) {

  ccs *At, *Att;
  if (!(At = transpose(A, p))) return NULL;

  if (!(Att = transpose(At, p))) { free_ccs(At); return NULL; }

  free_ccs(At);
  return Att;
}

int iperm(int n, int_t *order, int_t *iorder) {

  int i = 0;
  for (i=0; i<n; i++) {
    if (order[i] >= n || order[i] < 0)
      return -1;

    iorder[order[i]] = i;
  }

  for (i=0; i<n; i++) {
    if (iorder[order[i]] != i)
      return -1;
  }

  return 0;
}

ccs * chordalembedding(const ccs *X, int *nfill) {

  int i, j, k, n;
  ccs *A = chol_symbolic(X);
  if (!A) return NULL;

  n = A->ncols;
  *nfill = A->colptr[n] - ((X->colptr[n] - n)/2 + n);

  for (j=0; j<A->ncols; j++) {
    for (k=A->colptr[j], i=X->colptr[j]; k<A->colptr[j+1]; k++) {

      while (i<X->colptr[j+1] && X->rowind[i] < j) i++;

      if (i<X->colptr[j+1] && X->rowind[i] == A->rowind[k])
        ((double *)A->values)[k] = ((double *)X->values)[i++];
      else
        ((double *)A->values)[k] = 0;
    }
  }

  return A;
}

ccs * transpose(const ccs *A, int_t *p) {

  int_t i, j, *buf;
  ccs *B = alloc_ccs(A->ncols, A->nrows, A->colptr[A->ncols]);
  if (!B) return NULL;

  if (!(buf = calloc(A->nrows,sizeof(int_t)))) { free_ccs(B); return NULL; }

  /* Run through matrix and count number of elms in each row */
  if (p) {
    for (j=0; j<A->ncols; j++) {
      for (i=A->colptr[p[j]]; i<A->colptr[p[j]+1]-16; i += 16) {
        buf[ A->rowind[i] ]++;
        buf[ A->rowind[i+1] ]++;
        buf[ A->rowind[i+2] ]++;
        buf[ A->rowind[i+3] ]++;
        buf[ A->rowind[i+4] ]++;
        buf[ A->rowind[i+5] ]++;
        buf[ A->rowind[i+6] ]++;
        buf[ A->rowind[i+7] ]++;
        buf[ A->rowind[i+8] ]++;
        buf[ A->rowind[i+9] ]++;
        buf[ A->rowind[i+10] ]++;
        buf[ A->rowind[i+11] ]++;
        buf[ A->rowind[i+12] ]++;
        buf[ A->rowind[i+13] ]++;
        buf[ A->rowind[i+14] ]++;
        buf[ A->rowind[i+15] ]++;
      }
      for (; i<A->colptr[p[j]+1]; i++)
        buf[ A->rowind[i] ]++;

    }
  } else {
    for (j=0; j<A->ncols; j++) {
      for (i=A->colptr[j]; i<A->colptr[j+1]-16; i += 16) {
        buf[ A->rowind[i] ]++;
        buf[ A->rowind[i+1] ]++;
        buf[ A->rowind[i+2] ]++;
        buf[ A->rowind[i+3] ]++;
        buf[ A->rowind[i+4] ]++;
        buf[ A->rowind[i+5] ]++;
        buf[ A->rowind[i+6] ]++;
        buf[ A->rowind[i+7] ]++;
        buf[ A->rowind[i+8] ]++;
        buf[ A->rowind[i+9] ]++;
        buf[ A->rowind[i+10] ]++;
        buf[ A->rowind[i+11] ]++;
        buf[ A->rowind[i+12] ]++;
        buf[ A->rowind[i+13] ]++;
        buf[ A->rowind[i+14] ]++;
        buf[ A->rowind[i+15] ]++;
      }

      for (; i<A->colptr[j+1]; i++)
        buf[ A->rowind[i] ]++;
    }
  }

  /* generate new colptr */
  for (i=0; i<B->ncols; i++)
    B->colptr[i+1] = B->colptr[i] + buf[i];

  /* fill in rowind and values */
  for (i=0; i<A->nrows; i++) buf[i] = 0;

  for (j=0; j<A->ncols; j++) {
    for (i=A->colptr[p ? p[j] : j]; i<A->colptr[p ? p[j]+1 : j+1]-16; i += 16) {
      B->rowind[ B->colptr[A->rowind[i]] + buf[A->rowind[i]] ] = j;
      ((double *)B->values)[B->colptr[A->rowind[i]] + buf[A->rowind[i]]++] = ((double *)A->values)[i];

      B->rowind[ B->colptr[A->rowind[i+1]] + buf[A->rowind[i+1]] ] = j;
      ((double *)B->values)[B->colptr[A->rowind[i+1]] + buf[A->rowind[i+1]]++] = ((double *)A->values)[i+1];

      B->rowind[ B->colptr[A->rowind[i+2]] + buf[A->rowind[i+2]] ] = j;
      ((double *)B->values)[B->colptr[A->rowind[i+2]] + buf[A->rowind[i+2]]++] = ((double *)A->values)[i+2];

      B->rowind[ B->colptr[A->rowind[i+3]] + buf[A->rowind[i+3]] ] = j;
      ((double *)B->values)[B->colptr[A->rowind[i+3]] + buf[A->rowind[i+3]]++] = ((double *)A->values)[i+3];

      B->rowind[ B->colptr[A->rowind[i+4]] + buf[A->rowind[i+4]] ] = j;
      ((double *)B->values)[B->colptr[A->rowind[i+4]] + buf[A->rowind[i+4]]++] = ((double *)A->values)[i+4];

      B->rowind[ B->colptr[A->rowind[i+5]] + buf[A->rowind[i+5]] ] = j;
      ((double *)B->values)[B->colptr[A->rowind[i+5]] + buf[A->rowind[i+5]]++] = ((double *)A->values)[i+5];

      B->rowind[ B->colptr[A->rowind[i+6]] + buf[A->rowind[i+6]] ] = j;
      ((double *)B->values)[B->colptr[A->rowind[i+6]] + buf[A->rowind[i+6]]++] = ((double *)A->values)[i+6];

      B->rowind[ B->colptr[A->rowind[i+7]] + buf[A->rowind[i+7]] ] = j;
      ((double *)B->values)[B->colptr[A->rowind[i+7]] + buf[A->rowind[i+7]]++] = ((double *)A->values)[i+7];

      B->rowind[ B->colptr[A->rowind[i+8]] + buf[A->rowind[i+8]] ] = j;
      ((double *)B->values)[B->colptr[A->rowind[i+8]] + buf[A->rowind[i+8]]++] = ((double *)A->values)[i+8];

      B->rowind[ B->colptr[A->rowind[i+9]] + buf[A->rowind[i+9]] ] = j;
      ((double *)B->values)[B->colptr[A->rowind[i+9]] + buf[A->rowind[i+9]]++] = ((double *)A->values)[i+9];

      B->rowind[ B->colptr[A->rowind[i+10]] + buf[A->rowind[i+10]] ] = j;
      ((double *)B->values)[B->colptr[A->rowind[i+10]] + buf[A->rowind[i+10]]++] = ((double *)A->values)[i+10];

      B->rowind[ B->colptr[A->rowind[i+11]] + buf[A->rowind[i+11]] ] = j;
      ((double *)B->values)[B->colptr[A->rowind[i+11]] + buf[A->rowind[i+11]]++] = ((double *)A->values)[i+11];

      B->rowind[ B->colptr[A->rowind[i+12]] + buf[A->rowind[i+12]] ] = j;
      ((double *)B->values)[B->colptr[A->rowind[i+12]] + buf[A->rowind[i+12]]++] = ((double *)A->values)[i+12];

      B->rowind[ B->colptr[A->rowind[i+13]] + buf[A->rowind[i+13]] ] = j;
      ((double *)B->values)[B->colptr[A->rowind[i+13]] + buf[A->rowind[i+13]]++] = ((double *)A->values)[i+13];

      B->rowind[ B->colptr[A->rowind[i+14]] + buf[A->rowind[i+14]] ] = j;
      ((double *)B->values)[B->colptr[A->rowind[i+14]] + buf[A->rowind[i+14]]++] = ((double *)A->values)[i+14];

      B->rowind[ B->colptr[A->rowind[i+15]] + buf[A->rowind[i+15]] ] = j;
      ((double *)B->values)[B->colptr[A->rowind[i+15]] + buf[A->rowind[i+15]]++] = ((double *)A->values)[i+15];

    }

    for (; i<A->colptr[p ? p[j]+1 : j+1]; i++) {
      B->rowind[ B->colptr[A->rowind[i]] + buf[A->rowind[i]] ] = j;
      ((double *)B->values)[B->colptr[A->rowind[i]] + buf[A->rowind[i]]++] =
        ((double *)A->values)[i];
    }
  }

  free(buf);
  return B;
}

ccs * symmetrize(const ccs *A) {

  int j, k, cnt = 0;
  double *newval;
  int_t *newrow;
  ccs *At, *B;
  if (!(At = transpose(A, NULL))) return NULL;

  if (!(B = alloc_ccs(A->nrows, A->nrows, 0))) {
    free_ccs(At); free_ccs(B); return NULL;
  }

  for (j=0; j<B->ncols; j++)
    B->colptr[j+1] = B->colptr[j] +
    A->colptr[j+1]-A->colptr[j] + At->colptr[j+1]-At->colptr[j] -
    ((A->colptr[j+1]>A->colptr[j]) && (A->rowind[A->colptr[j]] == j));

  newval = malloc(B->colptr[B->ncols]*sizeof(double));
  newrow = malloc(B->colptr[B->ncols]*sizeof(int_t));
  if (!newval || !newrow) {
    free_ccs(At); free_ccs(B); free(newval); free(newrow); return NULL;
  }

  for (j=0; j<B->ncols; j++) {

    for (k=At->colptr[j]; k<At->colptr[j+1]; k++) {
      newrow[cnt] = At->rowind[k];
      newval[cnt] = ((double *)At->values)[k];
      cnt++;
    }

    for (k=A->colptr[j]; k<A->colptr[j+1]; k++) {
      if (A->rowind[k] > j) {
        newrow[cnt] = A->rowind[k];
        newval[cnt] = ((double *)A->values)[k];
        cnt++;
      }
    }
  }

  free(B->rowind);
  free(B->values);
  B->rowind = newrow;
  B->values = newval;

  free_ccs(At);
  return B;
}

ccs * tril(const ccs *A) {

  int j, k, cnt = 0;
  double *newval;
  int_t *newrow;
  ccs *B = alloc_ccs(A->nrows, A->nrows, 0);
  if (!B) return NULL;

  for (j=0; j<A->nrows; j++) {

    B->colptr[j+1] = B->colptr[j];
    for (k=A->colptr[j]; k<A->colptr[j+1]; k++)
      if (A->rowind[k] >= j) {
        B->colptr[j+1] += A->colptr[j+1]-k;
        break;
      }
  }

  newval = malloc(B->colptr[B->ncols]*sizeof(double));
  newrow = malloc(B->colptr[B->ncols]*sizeof(int_t));
  if (!newval || !newrow) {
    free_ccs(B); free(newval); free(newrow); return NULL;
  }

  for (j=0; j<A->ncols; j++) {

    for (k=A->colptr[j]; k<A->colptr[j+1]; k++) {
      if (A->rowind[k] >= j) {
        newrow[cnt] = A->rowind[k];
        newval[cnt] = ((double *)A->values)[k];
        cnt++;
      }

    }
  }

  free(B->rowind);
  free(B->values);
  B->rowind = newrow;
  B->values = newval;

  return B;
}

int istril(const ccs *A) {

  int j;

  for (j=0; j<A->nrows; j++) {
    if (A->colptr[j+1]-A->colptr[j] && A->rowind[A->colptr[j]] < j)
      return 0;
  }

  return 1;
}


/*
   Sparse accumulator (spa) - dense representation of a sparse vector
 */

spa * alloc_spa(int_t n) {

  int_t i;
  spa *s = malloc(sizeof(spa));

  if (s) {
    s->val = malloc( sizeof(double)*n );
    s->nz  = malloc( n*sizeof(char) );
    s->idx = malloc( n*sizeof(int) );
    s->nnz = 0;
    s->n = n;
  }

  if (!s || !s->val || !s->nz || !s->idx) {
    free(s->val);
    free(s->nz); free(s->idx); free(s);
    return NULL;
  }

  for (i=0; i<n; i++) s->nz[i] = 0;

  return s;
}

void free_spa(spa *s) {
  if (s) {
    free(s->val); free(s->nz); free(s->idx); free(s);
  }
}

void init_spa(spa *s, const ccs *X, int col) {
  int_t i;
  for (i=0; i<s->nnz; i++)
    s->nz[s->idx[i]] = 0;

  s->nnz = 0;

  if (X) {
    for (i=X->colptr[col]; i<X->colptr[col+1]; i++) {
      s->nz[X->rowind[i]] = 1;
      s->val[X->rowind[i]] = ((double *)X->values)[i];
      s->idx[s->nnz++] = X->rowind[i];
    }
  }
}

void spa_symb_axpy_uplo (const ccs *X, int col, spa *y, int j, char uplo) 
{
  int i;
  for (i=X->colptr[col]; i<X->colptr[col+1]; i++)
    if ((uplo == 'L' && X->rowind[i] >= j) ||
        (uplo == 'U' && X->rowind[i] <= j)) {

      if (!y->nz[X->rowind[i]]) {
        y->nz[X->rowind[i]] = 1;
        y->idx[y->nnz++] = X->rowind[i];
      }
    }
}

void spa2compressed(spa *s, ccs *A, int col) {

  int i, k=0;
  for (i=A->colptr[col]; i<A->colptr[col+1]; i++) {
      A->rowind[i] = s->idx[k];
      ((double *)A->values)[i] = s->val[s->idx[k++]];
  }
}

/* References:
 *
 * The Role of Elimination Trees, J. Liu,
 * SIAM J. Matrix Anal. Appl., 11 (1), pp. 134-172, Jan. 1990
 *
 * A Compact Row Storage Scheme for Cholesky Factors Using
 * Elimination Trees, J. Liu, ACM TOMS, 12 (2), June 1986
 */
void etree(const ccs *A, int_t *parent, int_t *work) {

  int i, k, r, n = A->nrows;
  int_t *ancestor = work;

  for (i=0; i<n; i++) {
    parent[i] = -1;
    ancestor[i] = -1;

    for (k=A->colptr[i]; k<A->colptr[i+1] && A->rowind[k] < i; k++) {

      r = A->rowind[k];
      while ((ancestor[r] != -1) && (ancestor[r] != i)) {
        int_t t = ancestor[r];
        ancestor[r] = i;
        r = t;
      }

      if (ancestor[r] == -1) {
        ancestor[r] = i;
        parent[r] = i;
      }
    }
  }
}

void chol_nz_count(const ccs *A, int_t *parent, int_t *work, int_t *colcnt) {

  int i, j, k, n = A->nrows;

  for (i=0; i<n; i++) {
    colcnt[i] = 1;
  }

  for (i=0; i<n; i++) {

    work[i] = i;
    for (k=A->colptr[i]; k<A->colptr[i+1] && A->rowind[k] < i; k++) {
      j = A->rowind[k];

      while (work[j] != i) {
        colcnt[j]++;
        work[j] = i;
        j = parent[j];
      }
    }
  }
}

/* A must be symmetric */
ccs * chol_symbolic(const ccs *A) {

  ccs *F, *Ft;
  spa *s;
  int n = A->nrows, i, k, nz = 0;
  int_t *parent, *work, *colcnt, *children_ptr, *children_cnt, *children_list;
  parent = malloc(4*n*sizeof(int_t));
  if (!parent) return NULL;

  work = parent + n;
  colcnt = parent + 2*n;

  etree(A, parent, work);
  chol_nz_count(A, parent, work, colcnt);

  for (i=0; i<n; i++)
    nz += colcnt[i];

  if (!(F = alloc_ccs(n, n, nz))) {
    free(parent);
    return NULL;
  }

  for (i=0; i<n; i++)
    F->colptr[i+1] = F->colptr[i] + colcnt[i];

  children_ptr  = parent + n;
  children_cnt  = parent + 2*n+1;
  children_list = parent + 3*n+1;

  for (i=0; i<3*n; i++)
    children_ptr[i] = 0;

  for (i=0; i<n; i++) {
    if (parent[i] > 0)
      children_ptr[1+parent[i]]++;
  }

//  printf("children_ptr: ");
  for (i=0; i<n; i++) {
    children_ptr[i+1] += children_ptr[i];
//    printf("%li ", children_ptr[i]);
  }
//  printf("\n");

  for (i=0; i<n; i++) {
    k = parent[i];
    if (k > 0) {
      children_list[ children_ptr[k] + children_cnt[k]++ ] = i;
    }
  }

//  printf("children_cnt: ");
//  for (i=0; i<n; i++) {
//    printf("%li ", children_cnt[i]);
//  }
//  printf("\n");
//
//  printf("children_list: ");
//  for (i=0; i<n-1; i++) {
//    printf("%li ", children_list[i]);
//  }
//  printf("\n");

  if (!(s = alloc_spa(n))) {
    free(parent);
    free_ccs(F);
    return NULL;
  }

  for (i=0; i<n; i++) {

    init_spa(s, 0, 0);
    spa_symb_axpy_uplo(A, i, s, i, 'L');

    /* add diagonal if it's missing */
    if (!s->nz[i]) {
      s->nz[i] = 1;
      s->idx[s->nnz++] = i;
    }

//    if (children_cnt[i])
//      printf("node %i has children: ", i);

    for (k=0; k<children_cnt[i]; k++) {
//      printf("%li ",children_list[children_ptr[i]+k]);
      spa_symb_axpy_uplo(F, children_list[children_ptr[i]+k],s,i,'L');
    }

//    if (children_cnt[i])
//      printf("\n");

    spa2compressed(s, F, i);
  }

  free_spa(s);
  free(parent);

  Ft = transpose(F, NULL);
  free_ccs(F);
  if (!Ft) return NULL;

  F = transpose(Ft, NULL);
  free_ccs(Ft);

  return F;
}
