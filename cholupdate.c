#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern void dtrsm_(char *side, char *uplo, char *transa, char *diag,
    int *m, int *n, double *alpha, double *A, int *lda, double *B,
    int *ldb);

extern void dgemv_(char* trans, int *m, int *n, double *alpha,
    const double *A, int *lda, const double *x, int *incx, double *beta, double *y, int *incy);

extern void dscal_(int *n, double *alpha, double *x, int *incx);

int chol_rank1_update(double *L, int n, int ld, double a, double *z, double *work) {

  double *w = work, *p = work + n, *Lk;
  char cL = 'L', cN = 'N';
  int i, k, ione = 1;
  double b, pk, r, done = 1.0;

  if (n == 0) return 0;

  /* w = z */
  memcpy(w, z, n*sizeof(double));

  /* Solve Lp = z */
  memcpy(p, z, n*sizeof(double));
  dtrsm_(&cL, &cL, &cN, &cN, &n, &ione, &done, L, &ld, p, &n);

  for (k=0; k<n; k++) {

    if (a==0.0) return 0;

    pk = p[k];
    r = 1 + a*pk*pk;
    if (r < 0.0) return -1;

    b = sqrt(r);
    Lk = &L[k*ld];

    for (i=k+1; i<n; i++)
      w[i] -= Lk[i]*pk;

    Lk[k] *= b;

    for (i=k+1; i<n; i++)
      Lk[i] = Lk[i]*b + w[i]*a*pk/b;

    a = a/(1 + a*pk*pk);
  }

  return 0;
}

int chol_add_row(double *L, int n, int ld, double *a, int k, double *work) {

  char cL = 'L', cN = 'N', cR = 'R', cT = 'T';
  int ione = 1, m;
  double done = 1.0, lkk, alpha, beta;

  int i, j, o1, o2;

  /* split L into   L :=  [L11, 0, 0;  0, 0, 0;  L31, 0, L33] */
  for (j=k-1; j>=0; j--) {
    o1 = j*ld;
    for (i=n-1; i>=k; i--)
      L[i+1 + o1] = L[i + o1];
  }

  for (j=n-1; j>=k; j--) {
    o1 = (j+1)*ld, o2 = j*ld;
    for (i=n-1; i>=k; i--) {
      L[i+1 + o1] = L[i + o2];
    }
  }

  for (j=0; j<k+1; j++)
    L[k + j*ld] = a[j];

  o1 = k*ld;
  for (j=k+1; j<n+1; j++)
    L[j + o1] = a[j];

  /* l21 = a21*L11^{-T} */
  dtrsm_(&cR, &cL, &cT, &cN, &ione, &k, &done, L, &ld, L + k, &ld);
  lkk = a[k];

  /* l22 = sqrt(l22 * nrm(l21)^2 */
  for (j=0; j<k; j++)
    lkk -= L[k + j*ld]*L[k + j*ld];

  if (lkk > 0.0)
    L[k + o1] = sqrt(lkk);
  else
    return -1;

  /* l32 = (a32 - L31*l12^T) */
  m = n-k;
  alpha = - 1/L[k + o1];
  beta = 1/L[k + o1];
  if (k>0)
    dgemv_(&cN, &m, &k, &alpha, L + k+1, &ld, L + k, &ld, &beta, L + k*ld + k+1, &ione);
  else {
    dscal_(&m, &beta, L + k*ld + k+1, &ione);
  }

  /* L33*L33.T = L33*L33.T - l32*l32.T */
  return chol_rank1_update(L + (k+1)*ld + k+1, m, ld, -1.0, L + k*ld + k+1, work);
}


int chol_delete_row(double *L, int n, int ld, int k, double *work) {

  int i, j, o1, o2, info;

  /* L33*L33.T = L33*L33.T + l32*l32.T */
  info = chol_rank1_update(L + (k+1)*ld + k+1, n-1-k, ld, 1.0, L + k*ld + k+1, work);
  if (info) return info;

  /* reduce L into   L :=  [L11, 0; L31, L33] */
  for (j=0; j<k; j++) {
    o1 = j*ld;
    for (i=k; i<n-1; i++)
      L[i + o1] = L[i+1 + o1];
  }

  for (j=k; j<n-1; j++) {
    o1 = j*ld, o2 = (j+1)*ld;
    for (i=k; i<n-1; i++) {
      L[i + o1] = L[i+1 + o2];
    }
  }

  return 0;
}
