#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "pf_cholesky.h"

int ldl_rank1_update(double *L, int n, int ld, double a, double *z, double *work, double droptol);

static void pdf_mf_update(pf_factor *f, int i, double droptol);

static void pdf_mf_lowrank_solve(pf_factor *f, int idx, chordalvec *B, int oB, int nB, char trans);

static int mf_solve(const chordalmatrix *R, chordalvec *X, int sys, double droptol);

static int cholesky_psd(chordalmatrix *X);

int ldl_simple(double *X, int ld, int n, double droptol);

static int ldl_solve(double *L, double *B, int m, int n, int sys, double droptol);

extern double dnrm2_(int *n, double *x, int *incx);

extern void dsyrk_(char *uplo, char *trans, int *n, int *k,
    double *alpha, double *A, int *lda, double *beta, double *B, int *ldb);

extern void dpotrf_(char *uplo, int *n, double *A, int *lda, int *info);

extern void dpotrs_(char *uplo, int *n, int *nrhs, double *A, int *lda,
    double *B, int *ldb, int *info);

extern void dtrtrs_(char *uplo, char *trans, char *diag, int *n, int *nrhs,
    double  *a, int *lda, double *b, int *ldb, int *info);

extern double ddot_(int *n, double *x, int *incx, double *y, int *incy);

extern void daxpy_(int *n, double *alpha, double *x, int *incx, double *y, int *incy);

extern void dgemm_(char *transa, char *transb, int *m, int *n, int *k,
    double *alpha, double *A, int *lda, double *B, int *ldb,
    double *beta, double *C, int *ldc);

extern void dtrsm_(char *side, char *uplo, char *transa, char *diag,
    int *m, int *n, double *alpha, double *A, int *lda, double *B, int *ldb);

#ifdef DLONG
extern int_t amd_l_order(int_t n, const int_t *Ap, const int_t *Ai, int_t *P, double *Control, double *Info);
extern void amd_l_defaults (double Control [ ]) ;
#define amd_order amd_l_order
#define amd_defaults amd_l_defaults
#else
extern int amd_order(int n, const int *Ap, const int *Ai, int *P, double *Control, double *Info);
extern void amd_defaults (double Control [ ]) ;
#endif

void free_pf_factor(pf_factor *f) {

  free(f->P);
  free(f->Pi);
  free(f->work2);
  if (f->p) chordalvec_destroy(f->p);
  if (f->beta) chordalvec_destroy(f->beta);
  if (f->work) chordalvec_destroy(f->work);
  if (f->A) chordalmatrix_destroy(f->A);
  if (f->Asymbolic) cliqueforest_destroy(f->Asymbolic);
  free(f);
}

pf_factor * pf_symbolic(const ccs *A, int_t *p, int ndense, double droptol, int output) {

  int i, nfill;
  ccs *Aperm, *Ae;
  pf_factor *f;
  int_t *p_ = malloc(A->nrows*sizeof(int_t));
  if (!p_) return NULL;

  if (p) {
    for (i=0; i<A->nrows; i++)
      p_[i] = p[i];
  }
  else {
    double Control [5], Info [20];
    amd_defaults(Control);
    amd_order(A->nrows, A->colptr, A->rowind, p_, Control, Info);
  }

  if (!(Aperm = perm(A, p_))) {
    free(p_);
    return NULL;
  }

  Ae = chol_symbolic(Aperm);
  nfill = Ae->colptr[Ae->ncols] - ((Aperm->colptr[A->ncols] - A->ncols)/2 + A->ncols);
  free_ccs(Aperm);
  if (!Ae) {
    free(p_);
    return NULL;
  }

  if (!(f = malloc(sizeof(pf_factor)))) {
    free(p_);
    free_ccs(Ae);
    return NULL;
  }
  f->A = NULL;
  f->P = p_;
  f->Pi = NULL;
  f->beta = NULL;
  f->p = NULL;
  f->work = NULL;
  f->work2 = NULL;
  f->n = A->nrows;
  f->ndense = ndense;
  f->droptol = droptol;
  f->is_numeric_factor = 0;

  if (cliqueforest_create(&f->Asymbolic, Ae, p_) != CHOMPACK_OK) {
    free_ccs(Ae);
    free_pf_factor(f);
    return NULL;
  }

  if (output) {
    printf("SYMBOLIC MF PDF FACTOR.\n");
    printf("SPARSE FACTOR: %i NONZEROS, %i FILL-IN ELEMENTS\n", (int)Ae->colptr[Ae->ncols], nfill);
  }

  free_ccs(Ae);

  if (!(f->work2 = calloc(3*f->n,sizeof(double)))) {
    free_pf_factor(f);
    return NULL;
  }

  if (!(f->Pi = malloc(f->n*sizeof(int_t)))) {
    free_pf_factor(f);
    return NULL;
  }

  for (i=0; i<f->n; i++)
    f->Pi[f->P[i]] = i;

  if (!(f->work = alloc_chordalvec(f->Asymbolic,1))) {
    free_pf_factor(f);
    return NULL;
  }

  if (!(f->p = alloc_chordalvec(f->Asymbolic,ndense)) ||
      !(f->beta = alloc_chordalvec(f->Asymbolic,ndense))) {
    free_pf_factor(f);
    return NULL;
  }

  return f;
}

int pf_numeric(pf_factor *f, const ccs *A, const double *B, char trans) {

  int i, j, res;
  ccs *Aperm = perm(A, f->P);
  if (!Aperm) return CHOMPACK_NOMEMORY;

  if (f->A) chordalmatrix_destroy(f->A);
  if (!(f->A = ccs_to_chordalmatrix(f->Asymbolic, Aperm))) {
    free_ccs(Aperm);
    return CHOMPACK_NOMEMORY;
  }
  free_ccs(Aperm);

  if ((res = cholesky_psd(f->A)))
    return res;

  // pi = P*B
  copy_dense_to_chordalvec(f->p, B, f->P, trans);

  // pi := L^{-1}*pi
  mf_solve(f->A, f->p, 0, f->droptol);
  for (i=0; i<f->ndense; i++) {

    for (j=0; j<i; j++) {
      // pi := Lj(betaj,pj)^{-1}*pi
      pdf_mf_lowrank_solve(f, j, f->p, i, 1, 'N');
    }
    pdf_mf_update(f, i, f->droptol);
  }

  f->is_numeric_factor = 1;
  return CHOMPACK_OK;
}

/*
 * Update the factor f->A := f->A + f->pi*f->pi.T
 */
static void pdf_mf_update(pf_factor *f, int i, double droptol) {

  cliqueforest *F = f->Asymbolic;
  int j, k, one = 1.0;

  double a = 1.0;

  for (k=F->nCliques-1; k>=0; k--) {

    int nk = nS(F,k);
    /* D[Sk,Sk] := D[Sk,Sk] + a*pi[Sk]*pi[Sk] */
    ldl_rank1_update(f->A->SS[k], nk, nk, a, f->p->S[k] + i*nk, f->work2, droptol);

    /* beta_k = a*inv(D[Sk,Sk])*p[Sk] */
    for (j=0; j<nk; j++)
      f->beta->S[k][i*nk + j] = a*f->p->S[k][i*nk + j];

    if (ldl_solve(f->A->SS[k], f->beta->S[k] + i*nk, nk, 1, 0, droptol)) {
      printf("XXX: LDL_SOLVE ERROR IN PDF_MF_UPDATE\n");
    }

    a *= 1 - ddot_(&nk, f->beta->S[k] + i*nk, &one, f->p->S[k] + i*nk, &one);
  }
}

/*
 * sys = 0: solve P*L*D*L'*P'*x = b
 * sys = 1: solve P*L*D^{1/2}*x = b
 * sys = 2: solve D^{1/2}*L'*P'*x = b
 */
int pf_solve(pf_factor *f, double *b, int n, int sys) {

  int i;
  chordalvec *bc = alloc_chordalvec(f->Asymbolic, n);
  if (!bc) return CHOMPACK_NOMEMORY;

  if (sys == 0 || sys == 1) {

    // solve P*L*L1*...*Lp*x = b
    copy_dense_to_chordalvec(bc, b, f->P, 'N');
    mf_solve(f->A, bc, 0, f->droptol);

    for (i=0; i<f->ndense; i++)
      pdf_mf_lowrank_solve(f, i, bc, 0, n, 'N');

    if (sys == 0) {
      // solve D*x = b
      mf_solve(f->A, bc, 2, f->droptol);

    } else {
      // solve D^{1/2}*x = b
      mf_solve(f->A, bc, 4, f->droptol);
      chordalvec_to_dense(bc, b, NULL);
    }
  }

  if (sys == 0 || sys == 2) {

    if (sys == 2) {
      copy_dense_to_chordalvec(bc, b, NULL, 'N');
      // solve D^{1/2}*x = b
      mf_solve(f->A, bc, 3, f->droptol);
    }

    // solve Lp^T*...*L1^T*L^T*P^T*x = b
    for (i=f->ndense-1; i>=0; i--)
      pdf_mf_lowrank_solve(f, i, bc, 0, n, 'T');

    mf_solve(f->A, bc, 1, f->droptol);
    chordalvec_to_dense(bc, b, f->P);
  }

  chordalvec_destroy(bc);
  return CHOMPACK_OK;
}


static void chordalvec_to_ccs(const chordalvec *X, ccs *Y, int coloffs, int n) {

  int i, j, k, l;
  cliqueforest *F = X->F;

  for (j=coloffs; j<coloffs+n; j++) {

    Y->colptr[j+1] += Y->colptr[j];

    for (k=0; k<F->nCliques; k++) {

      for (i=0, l=(j-coloffs)*nS(F,k); i<nS(F,k); i++) {
        double xij = X->S[k][i + l];
        if (xij != 0.0) {
          Y->rowind[Y->colptr[j+1]] = S(F,k)[i];
          ((double *)Y->values)[Y->colptr[j+1]++] = xij;
        }
      }
    }
  }
}

int pf_spsolve(pf_factor *f, ccs *b, ccs **x, int sys, int blksize) {

  int i, blkidx, maxnnz = 0;
  chordalvec *bc;
  spa *s = alloc_spa(b->nrows);
  if (!s) return CHOMPACK_NOMEMORY;

  if (!(bc = alloc_chordalvec(f->Asymbolic, MIN(b->ncols, blksize)))) {
    free_spa(s);
    return CHOMPACK_NOMEMORY;
  }

  for (blkidx=0; blkidx<b->ncols; blkidx += blksize) {

    int n = blkidx + blksize > b->ncols ? b->ncols - blkidx: blksize;

    if (blkidx == 0) {
      maxnnz = b->nrows*n;
      *x = alloc_ccs(b->nrows, b->ncols, maxnnz);
      if (!(*x)) {
        free_spa(s);
        chordalvec_destroy(bc);
        return CHOMPACK_NOMEMORY;
      }
    }

    if ((sys == 0) || (sys == 1)) {

      // solve P*L*L1*...*Lp*x = b
      sparse_to_chordalvec(bc, n, b, blkidx, s, f->P);
      mf_solve(f->A, bc, 0, f->droptol);

      for (i=0; i<f->ndense; i++)
        pdf_mf_lowrank_solve(f, i, bc, 0, n, 'N');

      if (sys == 0) {
        // solve D*x = b
        mf_solve(f->A, bc, 2, f->droptol);

      } else {
        // solve D^{1/2}*x = b
        mf_solve(f->A, bc, 4, f->droptol);

        chordalvec_to_ccs(bc, *x, blkidx, n);
        if ((blkidx + n < b->ncols) && ((*x)->colptr[blkidx+n] > maxnnz - blksize*b->nrows)) {
          maxnnz = 2*maxnnz;
          (*x)->values = realloc((*x)->values, maxnnz*sizeof(double));
          (*x)->rowind = realloc((*x)->rowind, maxnnz*sizeof(int_t));

          if ( (!(*x)->values) || (!(*x)->rowind) ) {
            free_spa(s);
            chordalvec_destroy(bc);
            return CHOMPACK_NOMEMORY;
          }
        }
      }
    }

    if (sys == 0 || sys == 2) {

      if (sys == 2) {
        sparse_to_chordalvec(bc, n, b, blkidx, s, 0);
        // solve D^{1/2}*x = b
        mf_solve(f->A, bc, 3, f->droptol);
      }

      // solve Lp^T*...*L1^T*L^T*P^T*x = b
      for (i=f->ndense-1; i>=0; i--)
        pdf_mf_lowrank_solve(f, i, bc, 0, n, 'T');

      mf_solve(f->A, bc, 1, f->droptol);

      chordalvec_to_ccs(bc, *x, blkidx, n);
      if ((blkidx + n < b->ncols) && ((*x)->colptr[blkidx+n] > maxnnz - blksize*b->nrows)) {
        maxnnz = 2*maxnnz;
        (*x)->values = realloc((*x)->values, maxnnz*sizeof(double));
        (*x)->rowind = realloc((*x)->rowind, maxnnz*sizeof(int_t));

        if ( (!(*x)->values) || (!(*x)->rowind) ) {
          free_spa(s);
          chordalvec_destroy(bc);
          return CHOMPACK_NOMEMORY;
        }
      }
    }

  }

  (*x)->values = realloc((*x)->values, (*x)->colptr[(*x)->ncols]*sizeof(double));
  (*x)->rowind = realloc((*x)->rowind, (*x)->colptr[(*x)->ncols]*sizeof(int_t));

  free_spa(s);
  chordalvec_destroy(bc);

  if ((sys == 0) || (sys == 2)) {
    ccs *xt = transpose(*x, 0);
    if (!xt) return CHOMPACK_NOMEMORY;
    free_ccs(*x);
    *x = xt;

    xt = transpose(*x, f->Pi);
    if (!(xt)) return CHOMPACK_NOMEMORY;
    free_ccs(*x);
    *x = xt;
  }

  return CHOMPACK_OK;
}

int ldl_simple(double *X, int ld, int n, double droptol) {

  int i, j, k;

  for (j=0; j<n-1; j++) {

    double *Xj  = X + j*ld;
    double  Xjj = Xj[j];

    if (Xjj > droptol) {

      for (i=j+1; i<n; i++) {
        Xj[i] /= Xjj;
      }

      for (k=j+1; k<n; k++) {
        double *Xk = X + k*ld;
        double g_jk = Xj[k]*Xjj;

        if (g_jk != 0.0) {
          for (i=k; i<n-16; i += 16) {
            Xk[i]    -= Xj[i]*g_jk;
            Xk[i+1]  -= Xj[i+1]*g_jk;
            Xk[i+2]  -= Xj[i+2]*g_jk;
            Xk[i+3]  -= Xj[i+3]*g_jk;
            Xk[i+4]  -= Xj[i+4]*g_jk;
            Xk[i+5]  -= Xj[i+5]*g_jk;
            Xk[i+6]  -= Xj[i+6]*g_jk;
            Xk[i+7]  -= Xj[i+7]*g_jk;
            Xk[i+8]  -= Xj[i+8]*g_jk;
            Xk[i+9]  -= Xj[i+9]*g_jk;
            Xk[i+10] -= Xj[i+10]*g_jk;
            Xk[i+11] -= Xj[i+11]*g_jk;
            Xk[i+12] -= Xj[i+12]*g_jk;
            Xk[i+13] -= Xj[i+13]*g_jk;
            Xk[i+14] -= Xj[i+14]*g_jk;
            Xk[i+15] -= Xj[i+15]*g_jk;
          }

          for (; i<n; i++) {
            Xk[i] -= Xj[i]*g_jk;
          }
        }
      }
    }
    else if (Xjj > -droptol) {
      Xj[j] = 0.0;
    }
    else return CHOMPACK_FACTORIZATION_ERR;
  }

  return CHOMPACK_OK;
}

int ldl(double *X, int ld, int n, int blksize, double droptol) {

  int i, j, k1, k2, offs, ione=1;
  char cL = 'L', cU = 'U', cT = 'T', cN = 'N', cR = 'R';
  double done = 1.0, dminusone = -1.0, v;

  for (k1 = 0; k1<n; k1 += blksize) {
    int mk, nk;
    k2 = k1+blksize < n ? k1+blksize : n;
    mk = n-k2;
    nk = k2-k1;

    /* X[k1:k2,k1:k2] = L*D*L' */
    if (ldl_simple(X + k1*ld + k1, ld, k2-k1, droptol))
      return CHOMPACK_FACTORIZATION_ERR;

    /* X[k2:n,k1:k2] := X[k2:n,k1:k2]*L^{-T}*D^{-1/2} */
    dtrsm_(&cR, &cL, &cT, &cU, &mk, &nk, &done, X + k1*ld + k1, &ld, X + k1*ld + k2, &ld);

    for (j=k1; j<k2; j++) {
      offs = j*ld;
      v = X[offs + j];
      if (v < -droptol)
        return CHOMPACK_FACTORIZATION_ERR;
      else if (v < droptol) {
        if (dnrm2_(&mk, X + offs + k2, &ione) > droptol) {
          return CHOMPACK_FACTORIZATION_ERR;
        }
      }
      else {
        v = sqrt(v);
        for (i=k2; i<n; i++)
          X[offs + i] /= v;
      }
    }

    /* X[k2:n,k2:n] := X[k2:n,k2:n] - X[k2:n,k1:k2]*X[k2:n,k1:k2]' */
    dsyrk_(&cL, &cN, &mk, &nk, &dminusone, X + k1*ld + k2, &ld, &done, X + k2*ld + k2, &ld);

    /* X[k2:n,k1:k2] := X[k2:n,k1:k2]*D^{-1/2} */
    for (j=k1; j<k2; j++) {
      offs = j*ld;
      v = X[offs + j];
      if (v < -droptol)
        return CHOMPACK_FACTORIZATION_ERR;
      else  {
        v = sqrt(v);
        for (i=k2; i<n; i++)
          X[offs + i] /= v;
      }
    }
  }

  return CHOMPACK_OK;
}

/*
 * X := L^{-T}*D^{-1}*L^{-1}*X   if sys = 0
 * X := L^{-T}*D^{-1/2}*X        if sys = 1
 * X := D^{-1/2}*L^{-1}*X        if sys = 2
 * X := X*L^{-T}*D^{-1}*L^{-1}   if sys = 3
 * X := X*L^{-T}*D^{-1/2}        if sys = 4
 * X := X*D^{-1/2}*L^{-1}        if sys = 5
 */
static int ldl_solve(double *L, double *B, int m, int n, int sys, double droptol) {

  int i, j, res = CHOMPACK_OK;
  char cL = 'L', cR = 'R', cU = 'U', cN = 'N', cT = 'T';
  double done = 1.0;

  if (sys == 0) { // X := L^{-T}*D^{-1}*L^{-1}*X

    dtrsm_(&cL, &cL, &cN, &cU, &m, &n, &done, L, &m, B, &m);

    for (i=0; i<m; i++) {

      double Lii = L[i + i*m];
      if (fabs(Lii) < droptol) {
        for (j=0; j<n; j++)
          if (fabs(B[i + j*m]) > droptol) {
            B[i + j*m] = 0;
            res = CHOMPACK_FACTORIZATION_ERR;
          }
      }
      else {
        for (j=0; j<n; j++)
          B[i + j*m] /= Lii;
      }
    }
    dtrsm_(&cL, &cL, &cT, &cU, &m, &n, &done, L, &m, B, &m);
  }
  else if (sys == 1) { // X := L^{-T}*D^{-1/2}*X

    for (i=0; i<m; i++) {

      double Lii = L[i + i*m];
      if (fabs(Lii) < droptol) {
        for (j=0; j<n; j++)
          if (fabs(B[i + j*m]) > droptol) {
            B[i + j*m] = 0;
            res = CHOMPACK_FACTORIZATION_ERR;
          }
      }
      else {
        Lii = sqrt(Lii);
        for (j=0; j<n; j++)
          B[i + j*m] /= Lii;
      }
    }
    dtrsm_(&cL, &cL, &cT, &cU, &m, &n, &done, L, &m, B, &m);
  }
  else if (sys == 2) { // X := D^{-1/2}*L^{-1}*X

    dtrsm_(&cL, &cL, &cN, &cU, &m, &n, &done, L, &m, B, &m);

    for (i=0; i<m; i++) {

      double Lii = L[i + i*m];
      if (fabs(Lii) < droptol) {
        for (j=0; j<n; j++)
          if (fabs(B[i + j*m]) > droptol) {
            B[i + j*m] = 0;
            res = CHOMPACK_FACTORIZATION_ERR;
          }
      }
      else {
        Lii = sqrt(Lii);
        for (j=0; j<n; j++)
          B[i + j*m] /= Lii;
      }
    }
  }
  else if (sys == 3) { // X := X*L^{-T}*D^{-1}*L^{-1}

    dtrsm_(&cR, &cL, &cT, &cU, &m, &n, &done, L, &n, B, &m);

    for (j=0; j<n; j++) {

      double Ljj = L[j + j*n];
      if (fabs(Ljj) < droptol) {
        for (i=0; i<m; i++)
          if (fabs(B[i + j*m]) > droptol) {
            B[i + j*m] = 0;
            res = CHOMPACK_FACTORIZATION_ERR;
          }
      }
      else {

        for (i=0; i<m; i++)
          B[i + j*m] /= Ljj;

      }
    }
    dtrsm_(&cR, &cL, &cN, &cU, &m, &n, &done, L, &n, B, &m);
  }
  else if (sys == 4) { // X := X*L^{-T}*D^{-1/2}

    dtrsm_(&cR, &cL, &cT, &cU, &m, &n, &done, L, &n, B, &m);

    for (j=0; j<n; j++) {

      double Ljj = L[j + j*n];
      if (fabs(Ljj) < droptol) {
        for (i=0; i<m; i++)
          if (fabs(B[i + j*m]) > droptol) {
            B[i + j*m] = 0;
            res = CHOMPACK_FACTORIZATION_ERR;
          }
      }
      else {
        Ljj = sqrt(Ljj);
        for (i=0; i<m; i++)
          B[i + j*m] /= Ljj;
      }
    }
  }
  else if (sys == 5) { // X := X*D^{-1/2}*L^{-1}

    for (j=0; j<n; j++) {

      double Ljj = L[j + j*n];
      if (fabs(Ljj) < droptol) {
        for (i=0; i<m; i++)
          if (fabs(B[i + j*m]) > droptol) {
            B[i + j*m] = 0;
            res = CHOMPACK_FACTORIZATION_ERR;
          }
      }
      else {
        Ljj = sqrt(Ljj);
        for (i=0; i<m; i++)
          B[i + j*m] /= Ljj;
      }
    }

    dtrsm_(&cR, &cL, &cN, &cU, &m, &n, &done, L, &n, B, &m);
  }

  return res;
}


static int cholesky_psd(chordalmatrix *X) {

  int l, k;
  char cL = 'L', cN = 'N';
  double alpha = -1.0, beta = 1.0;

  cliqueforest *F = X->F;

  for (k=F->nCliques-1; k>=0; k--) {

    /* allocate the X_{Uk,Uk} block; it will be freed by the parent clique */
    if (!(X->UU[k] = calloc(nU(F,k)*nU(F,k),sizeof(double)))) {
      for (l=F->nCliques-1; l>k; l--) FREE(X->UU[l]);
      return CHOMPACK_NOMEMORY;
    }

    for (l=0; l<F->list[k]->nchildren; l++)
      {
        int chidx = F->list[k]->children[l]->listidx;

        /*
         * Add the children X_{Uc,Uc} matrices (frontal matrices) to X_{Vk,Vk}
         * and free the X_{Uc,Uc} blocks, which are no longer needed.
         */
        cliquecopy(X, F->list[chidx], 1);
        FREE(X->UU[chidx]);
      }

    /* Factor X_{Sk,Sk} */
    if (ldl(X->SS[k], nS(F,k), nS(F,k), 32, 1e-14)) {
      for (l=F->nCliques-1; l>=k; l--) FREE(X->UU[l]);
      return CHOMPACK_FACTORIZATION_ERR;
    }

    if (nU(F,k)) {    /* if |U_k| > 0 */

      // X_US := X_US*L^{-T}*D^{-1/2}
      if (ldl_solve(X->SS[k], X->US[k], nU(F,k), nS(F,k), 4, 1e-14)) {
        for (l=F->nCliques-1; l>=k; l--) FREE(X->UU[l]);
        return CHOMPACK_FACTORIZATION_ERR;
      }

      /* X_{Uk,Uk} := X_{Uk,Uk} - X_{Uk,Sk} * X_{Sk,Sk}^{-1} * X_{Uk,Sk}' */
      dsyrk_(&cL, &cN, &nU(F,k), &nS(F,k), &alpha, X->US[k], &nU(F,k), &beta, X->UU[k], &nU(F,k));

      // X_US := X_US*D^{-1/2}*L^{-2}
      if (ldl_solve(X->SS[k], X->US[k], nU(F,k), nS(F,k), 5, 1e-14)) {
        for (l=F->nCliques-1; l>=k; l--) FREE(X->UU[l]);
        return CHOMPACK_FACTORIZATION_ERR;
      }
    }
  }

  return CHOMPACK_OK;
}

int ldl_rank1_update(double *L, int n, int ld, double a, double *z, double *work, double droptol) {

  char cL = 'L', cN = 'N', cU = 'U';
  int i, k, ione = 1;
  double done = 1.0;
  double *w = work, *p = work + n;

  if (n == 0) return 0;

  /* w = z */
  memcpy(w, z, n*sizeof(double));

  /* Solve Lp = z */
  memcpy(p, z, n*sizeof(double));
  dtrsm_(&cL, &cL, &cN, &cU, &n, &ione, &done, L, &ld, p, &n);

  for (k=0; k<n; k++) {

    double *Lk = &L[k*ld];
    double dk = Lk[k];
    double pk = p[k];
    double r = dk + a*pk*pk;

    if (fabs(a) < droptol) return 0;

    Lk[k] = r;

    for (i=k+1; i<n; i++)
      w[i] -= Lk[i]*pk;

    if (r > droptol) {
      double t = a*pk/r;
      for (i=k+1; i<n; i++)
        Lk[i] += w[i]*t;
    }
    else return -1;


    a = a*dk/(dk + a*pk*pk);
  }

  return 0;
}

static int iszero(double *x, int n) {

  int i;
  for (i=0; i<n; i++)
    if (x[i]) return 0;

  return 1;
}

/*
 * On exit, X is overwritten with
 * R*x = b    if sys = 0
 * R^T*x = b  if sys = 1
 * D*x = b    if sys = 2
 * Q*x = b,   if sys = 3
 * Q.T*x = b  if sys = 4
 * Q*Q.T = D
 */
static int mf_solve(const chordalmatrix *R, chordalvec *X, int sys, double droptol) {

  int k, l, n = X->ncols, res = CHOMPACK_OK;
  char cN = 'N', cT = 'T';
  double dminusone = -1.0, done = 1.0;

  cliqueforest *F = R->F;

  if (sys == 0) {   /* X := R^{-1} * X */

    for (k=F->nCliques-1; k>=0; k--) {

      if (nU(F,k)) {
        if (!(X->U[k] = calloc(nU(F,k)*n,sizeof(double)))) {
          return CHOMPACK_NOMEMORY;
        }
      }

      for (l=0; l<F->list[k]->nchildren; l++)
        {
          int chidx = F->list[k]->children[l]->listidx;
          cliquecopy_vec(X, F->list[chidx], 1);
          FREE(X->U[chidx]);
        }

      if (nU(F,k)) {

        if (!iszero(X->S[k], n*nS(F,k))) {
          dgemm_(&cN, &cN, &nU(F,k), &n, &nS(F,k),
              &dminusone, R->US[k], &nU(F,k), X->S[k], &nS(F,k),
              &done, X->U[k], &nU(F,k));
        }
      }
    }
  }

  else if (sys == 1) { /* X := R^{-T} * X */

    int *dealloc_cnt = malloc(F->nCliques*sizeof(int));

    for (k=0; k<F->nCliques; k++)
      dealloc_cnt[k] = F->list[k]->nchildren;

    for (k=0; k<F->nCliques; k++) {

      if (nU(F,k)) {

        if (!(X->U[k] = malloc(nU(F,k)*n*sizeof(double)))) {
          return CHOMPACK_NOMEMORY;
        }

        cliquecopy_vec(X, F->list[k], 0);

        if (--dealloc_cnt[F->list[k]->anc->listidx] == 0) {
          FREE(X->U[F->list[k]->anc->listidx]);
        }

        if (!iszero(X->U[k], n*nU(F,k))) {
          dgemm_(&cT, &cN, &nS(F,k), &n, &nU(F,k),
              &dminusone, R->US[k], &nU(F,k), X->U[k], &nU(F,k),
              &done, X->S[k], &nS(F,k));
        }

        if (F->list[k]->nchildren == 0) FREE(X->U[k]);
      }
    }

    free(dealloc_cnt);
  }

  else if (sys == 2) { /* X := D^{-1} * X */

    for (k=0; k<F->nCliques; k++) {

      if (ldl_solve(R->SS[k], X->S[k], nS(F,k), n, 0, droptol))
        res = CHOMPACK_SOLVE_ERR;
    }
  }

  else if (sys == 3) { /* X := Q^{-1} * X */

    for (k=0; k<F->nCliques; k++) {

      if (ldl_solve(R->SS[k], X->S[k], nS(F,k), n, 1, droptol))
        res = CHOMPACK_SOLVE_ERR;

    }
  }

  else if (sys == 4) { /* X := Q^{-T} * X */

    for (k=0; k<F->nCliques; k++) {

      if (ldl_solve(R->SS[k], X->S[k], nS(F,k), n, 2, droptol))
        res = CHOMPACK_SOLVE_ERR;
    }
  }

  return res;
}

static void pdf_mf_lowrank_solve(pf_factor *f, int idx, chordalvec *B, int oB, int nB, char trans) {

  int i, k, one=1;
  double dzero = 0.0, done = 1.0, dminusone = -1.0;
  cliqueforest *F = f->Asymbolic;

  char cN = 'N';
  if (trans=='N') {

    for (k=F->nCliques-1; k>=0; k--) {

      int nk = nS(F,k);
      // work2 := beta[Sk,idx].T*b[Sk,oB:oB+nB]
      dgemm_(&cN, &cN, &one, &nB, &nk, &done, f->beta->S[k] + idx*nk, &one, B->S[k] + oB*nk, &nk, &dzero, f->work2, &one);

      for (i=0; i<k; i++) {

        // b[Si,oB:oB:nB] -= p[Si, idx]*work2
        int ni = nS(F,i);
        dgemm_(&cN, &cN, &ni, &nB, &one, &dminusone, f->p->S[i] + idx*ni, &ni, f->work2, &one, &done, B->S[i] + oB*ni, &ni);
      }
    }

  } else {

    for (k=0; k<F->nCliques; k++) {

      int nk = nS(F,k);
      // work2 := p[Sk, idx].T*b[Sk,oB:oB+nB]
      dgemm_(&cN, &cN, &one, &nB, &nk, &done, f->p->S[k] + idx*nk, &one, B->S[k] + oB*nk, &nk, &dzero, f->work2, &one);

      for (i=k+1; i<F->nCliques; i++) {

        // b[Si,oB:oB:nB] -= beta[Si, idx]*work2
        int ni = nS(F,i);
        dgemm_(&cN, &cN, &ni, &nB, &one, &dminusone, f->beta->S[i] + idx*ni, &ni, f->work2, &one, &done, B->S[i] + oB*ni, &ni);

      }
    }
  }
}
