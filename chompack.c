#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "chompack.h"
#include "cliquetree.h"
#include "cholupdate.h"

extern void daxpy_(int *n, double *alpha, double *x, int *incx,
    double *y, int *incy);
extern double ddot_(int *n, double *x, int *incx, double *y, int *incy);

extern void dtrsm_(char *side, char *uplo, char *transa, char *diag,
    int *m, int *n, double *alpha, double *A, int *lda, double *B,
    int *ldb);
extern void dtrmm_(char *side, char *uplo, char *transa, char *diag,
    int *m, int *n, double *alpha, double *A, int *lda, double *B,
    int *ldb);
extern void dsyrk_(char *uplo, char *trans, int *n, int *k,
    double *alpha, double *A, int *lda, double *beta, double *B,
    int *ldb);
extern void dsyr2k_(char *uplo, char *trans, int *n, int *k,
    double *alpha, double *A, int *lda, double *B, int *ldb,
    double *beta, double *C, int *ldc);
extern void dsymm_(char *side, char *uplo, int *m, int *n,
    double *alpha, double *A, int *lda, double *B, int *ldb,
    double *beta, double *C, int *ldc);
extern void dgemm_(char *transa, char *transb, int *m, int *n, int *k,
    double *alpha, double *A, int *lda, double *B, int *ldb,
    double *beta, double *C, int *ldc);

extern void dpotrf_(char *uplo, int *n, double *A, int *lda, int *info);
extern void dpotrs_(char *uplo, int *n, int *nrhs, double *A, int *lda,
    double *B, int *ldb, int *info);
extern void dtrtri_(char *uplo, char *diag, int *n, double  *a, int *lda,
    int *info);
extern void dposv_(char *uplo, int *n, int *nrhs, double *A, int *lda,
    double *B, int *ldb, int *info);

/*
 *  Overwrites A := A*A' or A := A'*A where A is a lower-triangular matrix
 */
static void trmtrm(double *A, char trans, int n) {

  int j;

  char side, uplo = 'L', transA, diag = 'N';
  double alpha = 1.0;
  int lda = n, one = 1, n_;

  if (trans == 'N') {

    side = 'R'; transA = 'N';
    for (j=n-1; j>=0; j--) {
      n_ = j + 1;
      A[j + j*n] = ddot_(&n_, A + j, &lda, A + j, &lda);

      dtrmm_(&side, &uplo, &transA, &diag, &one, &j, &alpha, A,  &lda, A + j, &lda);
    }
  } else {

    side = 'L'; transA = 'T';
    for (j=0; j<n; j++) {
      n_ = n-j;
      A[j + j*n] = ddot_(&n_, A + j + j*n, &one, A + j + j*n, &one);

      n_ = n-j-1;
      dtrmm_(&side, &uplo, &transA, &diag, &n_, &one, &alpha, A + j+1 + (j+1)*n,  &lda, A + j+1 + j*n, &lda);
    }
  }
}

static void dtril(double *X, double *Y, int n) {
  int i, j;

  for (j=0; j<n; j++) {

    if (X != Y) for (i=0; i<=j; i++) Y[i*n + j] = X[i*n + j];

    for (i=j+1; i<n; i++) Y[i*n + j] = 0.0;
  }
}

static void unpack(double *X, int n) {

  int i, j;

  for (j=0; j<n; j++)
    for (i=0; i<j; i++)
      X[i + j*n] = X[j + i*n];
}

int llt(chordalmatrix *X) {

  int k;
  char cL = 'L', cR = 'R', cT = 'T', cN = 'N';
  double one = 1.0, zero = 0.0;

  cliqueforest *F = X->F;

  for (k=0; k<F->nCliques; k++) {

    double *tSS = malloc(nS(F,k)*nS(F,k)*sizeof(double));
    if (!tSS) return CHOMPACK_NOMEMORY;

    dtril(X->SS[k], tSS, nS(F,k));

    dsyrk_(&cL, &cN, &nS(F,k), &nS(F,k),
        &one, tSS, &nS(F,k), &zero, X->SS[k], &nS(F,k));

    if (nU(F,k)) {
      X->UU[k] = malloc(nU(F,k)*nU(F,k)*sizeof(double));
      if (!X->UU[k]) {
        free(tSS);
        return CHOMPACK_NOMEMORY;
      }

      dtrmm_(&cR, &cL, &cN, &cN,
          &nU(F,k), &nS(F,k), &one, tSS, &nS(F,k), X->US[k], &nU(F,k));

      dsyrk_(&cL, &cN, &nU(F,k), &nS(F,k),
          &one, X->US[k], &nU(F,k), &zero, X->UU[k], &nU(F,k));

      dtrmm_(&cR, &cL, &cT, &cN,
          &nU(F,k), &nS(F,k), &one, tSS, &nS(F,k), X->US[k], &nU(F,k));

      cliquecopy_nmf(X, F->list[k], 1, NULL);

      FREE(X->UU[k]);
    }
    free(tSS);
  }
  return CHOMPACK_OK;
}

///*
// * Parallel Cholesky solver for a clique-tree
// */
//static int cholesky_recursive(chordalmatrix *X, clique *node) {
//
//  int l, k = node->listidx, info = 0;
//  char cL = 'L', cR = 'R', cT = 'T', cN = 'N';
//  double alpha = 1.0, beta = 1.0;
//
//  cliqueforest *F = X->F;
//
//  /* factor the children cliques */
//#pragma omp parallel for shared(X, F, k, info), private(l)
//  for (l=0; l<F->list[k]->nchildren; l++)
//    {
//      int chidx = F->list[k]->children[l]->listidx;
//      int res = cholesky_recursive(X, F->list[chidx]);
//      if (res != CHOMPACK_OK) info = res;
//    }
//  if (info) return CHOMPACK_FACTORIZATION_ERR;
//
//  /* allocate the X_{Uk,Uk} block; it will be freed by the parent clique */
//  if ( (nU(F,k)) && !(X->UU[k] = calloc(nU(F,k)*nU(F,k),sizeof(double))) )
//      return CHOMPACK_NOMEMORY;
//
//  for (l=0; l<F->list[k]->nchildren; l++)
//    {
//      /*
//       * Add the children X_{Uc,Uc} matrices (frontal matrices) to X_{Vk,Vk}
//       * and free the X_{Uc,Uc} blocks, which are no longer needed.
//       */
//      int chidx = F->list[k]->children[l]->listidx;
//      cliquecopy(X, F->list[chidx], 1);
//      FREE(X->UU[chidx]);
//    }
//
//  /* Factor X_{Sk,Sk} */
//  dpotrf_(&cL,&nS(F,k),X->SS[k],&nS(F,k),&info);
//  if (info) return CHOMPACK_FACTORIZATION_ERR;
//
//  if (nU(F,k)) {     /* if |U_k| > 0 */
//
//    /* X_{Uk,Sk} := X_{Uk,Sk} * X_{Sk,Sk}^{-1} */
//    dtrsm_(&cR, &cL, &cT, &cN, &nU(F,k), &nS(F,k), &alpha, X->SS[k], &nS(F,k), X->US[k], &nU(F,k));
//    dtrsm_(&cR, &cL, &cN, &cN, &nU(F,k), &nS(F,k), &alpha, X->SS[k], &nS(F,k), X->US[k], &nU(F,k));
//
//    /* X_{Uk,Uk} := X_{Uk,Uk} - X_{Uk,Sk}*inv(X_{Sk,Sk})*X_{Uk,Sk}' */
//    double *T = malloc(nU(F,k)*nS(F,k)*sizeof(double));
//    if (!T) return CHOMPACK_NOMEMORY;
//    memcpy(T, X->US[k], nU(F,k)*nS(F,k)*sizeof(double));
//
//    dtrmm_(&cR, &cL, &cN, &cN, &nU(F,k), &nS(F,k), &alpha, X->SS[k], &nS(F,k), T, &nU(F,k));
//
//    alpha = -1.0;
//    dsyrk_(&cL, &cN, &nU(F,k), &nS(F,k), &alpha, T, &nU(F,k), &beta, X->UU[k], &nU(F,k));
//    free(T);
//  }
//  return CHOMPACK_OK;
//}
//
///*
// * Parallel Cholesky solver for a clique-forest. Each subtree is factored
// * by a call to cholesky_recursive()
// */
//int cholesky_parallel(chordalmatrix *X) {
//
//  int k, r;
//  for (k=0; k<X->F->nRoots; k++) {
//
//    r = cholesky_recursive(X, X->F->roots[k]);
//    if (r != CHOMPACK_OK) {
//      for (k=0; k<X->F->nCliques; k++)
//        FREE(X->UU[k]);
//
//      return r;
//    }
//  }
//
//  return CHOMPACK_OK;
//}

/*
 * Sequential Cholesky factorization
 */
int cholesky(chordalmatrix *X) {

  int l, k, info;
  char cL = 'L', cR = 'R', cT = 'T', cN = 'N';
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
    dpotrf_(&cL,&nS(F,k),X->SS[k],&nS(F,k),&info);
    if (info) {
      for (l=F->nCliques-1; l>=k; l--) FREE(X->UU[l]);
      return CHOMPACK_FACTORIZATION_ERR;
    }

    if (nU(F,k)) {    /* if |U_k| > 0 */

      /* X_{Uk,Sk} := X_{Uk,Sk}*X_{Sk,Sk}^{-1/2} */
      dtrsm_(&cR, &cL, &cT, &cN,
          &nU(F,k), &nS(F,k), &alpha, X->SS[k], &nS(F,k), X->US[k], &nU(F,k));

      /* X_{Uk,Uk} := X_{Uk,Uk} - X_{Uk,Sk} * X_{Sk,Sk}^{-1} * X_{Uk,Sk}' */
      dsyrk_(&cL, &cN, &nU(F,k), &nS(F,k),
          &alpha, X->US[k], &nU(F,k), &beta, X->UU[k], &nU(F,k));

      /* X_{Uk,Sk} := X_{Uk,Sk}*X_{Sk,Sk}^{-1/2} */
      dtrsm_(&cR, &cL, &cN, &cN,
          &nU(F,k), &nS(F,k), &alpha, X->SS[k], &nS(F,k), X->US[k], &nU(F,k));
    }
  }

  return CHOMPACK_OK;
}

static int update_UU_factor(const chordalmatrix *X, int k, double *z, double *work) {

  int i, j, l, n, info = 0, o1, o2;
  char cL = 'L';

  cliqueforest *F = X->F;
  clique *anc = F->list[k]->anc;
  clique *child = F->list[k];

  if (!anc->nu) {

    /* there is no parent Xuu block to downdate, so we form entire Xuu at once */
    double *Xuu = X->UU[k];
    double *Xss = X->SS[anc->listidx];
    for (j=0; j<child->nu; j++) {

      /* copy X_uu from ancestor X_ss */
      l = j;
      o1 = j*child->ldu;
      o2 = child->Uidx[j]*anc->ns;
      for (i=child->Uidx[j]; (i<anc->ns && l < child->nu); i++) {
        if (anc->S[i] == child->U[l]) {
          Xuu[l + o1] = Xss[i + o2];
          l++;
        }
      }
    }
    dpotrf_(&cL, &child->nu, X->UU[k], &child->ldu, &info);
    if (info)
      return -1;

  } else {

    /* downdate parent Xuu block */
    for (j=0; j<anc->nu; j++) {
      for (i=j; i<anc->nu; i++) {
        X->UU[k][i + j*child->ldu] = X->UU[anc->listidx][i + j*anc->ldu];
      }
    }

    n = anc->nu;
    for (j=child->nu-1, l=anc->nu-1; (j>=0 && l>=0); l--) {

      if ( anc->U[l] > child->U[j] ) {
        if (chol_delete_row(X->UU[k], n, child->ldu, l, work))
          return -1;

        n--;
      }
      else
        j--;
    }

    for (j=child->Uidx_ns-1; j>=0; j--) {

      /* copy X_{:,uj} from ancestor into z */
      l = 0;
      for (i=0; (i<anc->ns && l<child->nu); i++) {
        if (anc->S[i] == child->U[l]) {
          z[l] = X->SS[anc->listidx][child->Uidx[j]*anc->ns + i];
          l++;
        }
      }

      for (i=0; (i<anc->nu && l<child->nu); i++) {
        if (anc->U[i] == child->U[l]) {
          z[l] = X->US[anc->listidx][child->Uidx[j]*anc->nu + i];
          l++;
        }
      }

      if (chol_add_row(X->UU[k], n, child->ldu, z + j, 0, work))
        return -1;

      n += 1;
    }
  }
  return 0;
}

int partial_inv(chordalmatrix *X)
{
  int k, l, info;
  char cL = 'L', cT = 'T', cN = 'N';
  double one = 1.0, minusone = -1.0;

  cliqueforest *F = X->F;

  /* allocate temporary workspace */
  int *dealloc_cnt = malloc(F->nCliques*sizeof(int));
  double *work = malloc(3*F->n*sizeof(double));
  double *z = work + 2*F->n;
  if (!work || !dealloc_cnt) {
    free(work); free(dealloc_cnt);
    return CHOMPACK_NOMEMORY;
  }

  for (k=0; k<F->nCliques; k++)
    dealloc_cnt[k] = F->list[k]->nchildren;

  for (k=0; k<F->nCliques; k++) {

    X->UU[k] = malloc(F->list[k]->ldu*F->list[k]->ldu*sizeof(double));
    if (!X->UU[k]) {
      free(dealloc_cnt); free(work);
      for (l=0; l<=k; l++) FREE(X->UU[l]);
      return CHOMPACK_NOMEMORY;
    }

    if (nU(F,k)) {

      if (update_UU_factor(X, k, z, work)) {
        free(dealloc_cnt); free(work);
        for (l=0; l<=k; l++) FREE(X->UU[l]);
        return CHOMPACK_FACTORIZATION_ERR;
      }

      if (--dealloc_cnt[F->list[k]->anc->listidx] == 0) {

        FREE(X->UU[F->list[k]->anc->listidx]);
      }

      /* X_{Uk,Sk} := Q^T * X_{Uk,Sk} */
      dtrmm_(&cL, &cL, &cT, &cN,
          &nU(F,k), &nS(F,k), &one, X->UU[k], &F->list[k]->ldu, X->US[k], &nU(F,k));
    }

    /* X_{Sk,Sk} := X_{Sk,Sk}^{-1} + X_{Uk,Sk}^T * X_{Uk,Sk} */
    dtrtri_(&cL, &cN, &nS(F,k), X->SS[k], &nS(F,k), &info);
    trmtrm(X->SS[k], 'T', nS(F,k));

    if (info) {
      free(dealloc_cnt); free(work);
      for (l=0; l<=k; l++) FREE(X->UU[l]);
      return CHOMPACK_FACTORIZATION_ERR;
    }

    if (nU(F,k)) {
      dsyrk_(&cL, &cT, &nS(F,k), &nU(F,k),
          &one, X->US[k], &nU(F,k), &one, X->SS[k], &nS(F,k));

      /* X_{Uk,Sk} := -Q * X_{Uk,Sk} */
      dtrmm_(&cL, &cL, &cN, &cN,
          &nU(F,k), &nS(F,k), &minusone, X->UU[k], &F->list[k]->ldu, X->US[k], &nU(F,k));
    }

    if (F->list[k]->nchildren == 0) {
      FREE(X->UU[k]);
    }
  }

  free(dealloc_cnt);
  free(work);
  return CHOMPACK_OK;
}

chordalmatrix * completion(const chordalmatrix *C, int *info)
{
  int k, l;
  char cL = 'L', cT = 'T', cN = 'N';
  double one = 1.0, minusone = -1.0;

  cliqueforest  *F = C->F;

  /* allocate temporary workspace */
  int *dealloc_cnt = malloc(F->nCliques*sizeof(int));
  double *work = malloc(3*F->n*sizeof(double));
  double *z = work + 2*F->n;
  chordalmatrix *X;
  if (!work || !dealloc_cnt) {
    free(work); free(dealloc_cnt);
    *info = CHOMPACK_NOMEMORY;
    return NULL;
  }

  if (!(X = chordalmatrix_copy(C))) {
    free(dealloc_cnt); free(work);
    *info = CHOMPACK_NOMEMORY;
    return NULL;
  }

  for (k=0; k<F->nCliques; k++)
    dealloc_cnt[k] = F->list[k]->nchildren;

  for (k=0; k<F->nCliques; k++) {

    C->UU[k] = malloc(F->list[k]->ldu*F->list[k]->ldu*sizeof(double));
    if (!C->UU[k]) {
      free(dealloc_cnt); free(work);
      for (l=0; l<=k; l++) FREE(C->UU[l]);
      chordalmatrix_destroy(X);
      *info = CHOMPACK_NOMEMORY;
      return NULL;
    }

    if (nU(F,k)) {

      if (update_UU_factor(C, k, z, work)) {
        free(dealloc_cnt); free(work);
        for (l=0; l<=k; l++) FREE(C->UU[l]);
        chordalmatrix_destroy(X);
        *info = CHOMPACK_FACTORIZATION_ERR;
        return NULL;
      }

      if (--dealloc_cnt[F->list[k]->anc->listidx] == 0) {
        FREE(C->UU[F->list[k]->anc->listidx]);
      }

      /* X_{Uk,Sk} := Q^{-1} * C_{Uk,  Sk} */
      dtrsm_(&cL, &cL, &cN, &cN, &nU(F,k), &nS(F,k),
          &one, C->UU[k], &F->list[k]->ldu, X->US[k], &nU(F,k));

      /* X_{Sk,Sk} := (X_{Sk,Sk} - X_{Uk,Sk}^T * X_{Uk,Sk})^{-1} = L * L^T */
      dsyrk_(&cL, &cT, &nS(F,k), &nU(F,k),
          &minusone, X->US[k], &nU(F,k), &one, X->SS[k], &nS(F,k));
    }

    dpotrf_(&cL, &nS(F,k), X->SS[k], &nS(F,k), info);
    if (*info) {
      free(dealloc_cnt); free(work);
      for (l=0; l<=k; l++) FREE(C->UU[l]);
      chordalmatrix_destroy(X);
      *info = CHOMPACK_FACTORIZATION_ERR;
      return NULL;
    }

    dtrtri_(&cL, &cN, &nS(F,k), X->SS[k], &nS(F,k), info);
    if (*info) {
      free(dealloc_cnt); free(work);
      for (l=0; l<=k; l++) FREE(C->UU[l]);
      chordalmatrix_destroy(X);
      *info = CHOMPACK_FACTORIZATION_ERR;
      return NULL;
    }

    trmtrm(X->SS[k], 'T', nS(F,k));
    dpotrf_(&cL, &nS(F,k), X->SS[k], &nS(F,k), info);

    /* X_{Uk,Sk} := -Q^{-T} * X_{Uk,Sk} */
    if (nU(F,k))
      dtrsm_(&cL, &cL, &cT, &cN, &nU(F,k), &nS(F,k),
          &minusone, C->UU[k], &F->list[k]->ldu, X->US[k], &nU(F,k));

    if (F->list[k]->nchildren == 0) FREE(C->UU[k]);
  }

  free(work);
  free(dealloc_cnt);
  *info = CHOMPACK_OK;
  return X;
}


int hessian_factor
(const chordalmatrix *R, const chordalmatrix *Y, chordalmatrix **dX, int adj, int inv, int m)
{
  cliqueforest *F = R->F;
  int j, k, thread, nthreads, numfact, fact_error = CHOMPACK_OK, maxUS = 0;
  double *tUS, *work;
  int *dealloc_cnt;
  chordalmatrix **Ycopy;

  //
  // XXX:  CHECK DEALLOCATION IF FACTORIZATION FAILS
  //

#ifdef _OPENMP
  nthreads = omp_get_max_threads();
#else
  nthreads = 1;
#endif

  /* allocate temporary workspace */
  for (k=0; k<F->nCliques; k++) {
    maxUS = MAX(nU(F,k)*nS(F,k), maxUS);
  }
  tUS = malloc(nthreads*maxUS*sizeof(double));
  work = malloc(nthreads*3*F->n*sizeof(double));
  dealloc_cnt = malloc(nthreads*F->nCliques*sizeof(int));
  Ycopy = malloc(nthreads*sizeof(chordalmatrix *));
  if (!tUS || !dealloc_cnt || !work) {
    free(tUS); free(dealloc_cnt); free(work); free(Ycopy);
    return CHOMPACK_NOMEMORY;
  }

  for (k=0; k<nthreads-1; k++) {
    if (!(Ycopy[k] = chordalmatrix_copy(Y))) {
      for (j=0; j<k; j++)
        chordalmatrix_destroy(Ycopy[j]);

      free(tUS); free(dealloc_cnt); free(work); free(Ycopy);
      return CHOMPACK_NOMEMORY;
    }
  }
  Ycopy[nthreads-1] = (chordalmatrix *)Y;

#pragma omp parallel for \
        shared(R, Ycopy, dX, F, work, tUS, dealloc_cnt, maxUS, fact_error) \
          private(numfact, k, thread)

  for (numfact = 0; numfact < m; numfact++) {

    char cL = 'L', cR = 'R', cT = 'T', cN = 'N';
    double minusone = -1.0, one = 1.0;
    double *z;

#ifdef _OPENMP
    thread = omp_get_thread_num();
#else
    thread = 0;
#endif

    z = work + 2*F->n + thread*3*F->n;

    if (!inv) {
      /* Factor Hessian */
      if (!adj) {

        for (k=F->nCliques-1; k>=0; k--) {

          int l;
          /* allocate the dX_{Uk,Uk} block; it will be freed by the parent clique */
          if (nU(F,k))  dX[numfact]->UU[k] = calloc(nU(F,k)*nU(F,k),sizeof(double));

          for (l=0; l<F->list[k]->nchildren; l++)
            {
              int chidx = F->list[k]->children[l]->listidx;

              /*
               * Add the children L_{Uc,Uc} matrices (frontal matrices) to X_{Vk,Vk}
               * and free the X_{Uc,Uc} blocks, which are no longer needed.
               */
              cliquecopy(dX[numfact], F->list[chidx], 1);
              FREE(dX[numfact]->UU[chidx]);
            }

          if (nU(F,k)) {

            int n = nU(F,k)*nS(F,k), inc = 1;
            double zero = 0.0;

            /*  dX_{Uk,Uk} := dX_{Uk,Uk} + R_{Uk,Sk}*dX_{Sk,Sk}*R_{Uk,Sk}'
                  -(R_{Uk,Sk}*dX_{Uk,Sk}' + dX_{Uk,Sk}*R_{Uk,Sk}')      */
            /* dX_{Uk,Sk} := dX_{Uk,Sk} - R_{Uk,Sk}*dX_{Sk,Sk} */
            dsyr2k_(&cL, &cN, &nU(F,k), &nS(F,k), &minusone, R->US[k], &nU(F,k),
                dX[numfact]->US[k], &nU(F,k), &one, dX[numfact]->UU[k], &nU(F,k));

            dsymm_(&cR, &cL, &nU(F,k), &nS(F,k), &one, dX[numfact]->SS[k], &nS(F,k),
                R->US[k], &nU(F,k), &zero, tUS + thread*maxUS, &nU(F,k));

            dgemm_(&cN, &cT, &nU(F,k), &nU(F,k), &nS(F,k), &one, tUS + thread*maxUS,
                &nU(F,k), R->US[k], &nU(F,k), &one, dX[numfact]->UU[k], &nU(F,k));

            daxpy_(&n, &minusone, tUS + thread*maxUS, &inc, dX[numfact]->US[k], &inc);
          }
        }

        for (k=0; k<F->nCliques; k++)
          dealloc_cnt[thread*F->nCliques + k] = F->list[k]->nchildren;

        for (k=0; k<F->nCliques; k++) {

          /* dX_{Sk,Sk} := P_{k,k}^{-1} * dX_{Sk,Sk} * P_{k,k}^{-T},
                D_{Sk,Sk} = P_{k,k}*P_{k,k}^T */
          unpack(dX[numfact]->SS[k], nS(F,k));

          dtrsm_(&cL, &cL, &cN, &cN, &nS(F,k), &nS(F,k), &one, R->SS[k], &nS(F,k),
              dX[numfact]->SS[k], &nS(F,k));

          dtrsm_(&cR, &cL, &cT, &cN, &nS(F,k), &nS(F,k), &one, R->SS[k], &nS(F,k),
              dX[numfact]->SS[k], &nS(F,k));

          /*  dX_{Uk,Sk} = Q_k^T * dX_{Uk,Sk} * P_k^{-T},  Yuu = Q * Q^T */
          if (nU(F,k)) {

            if (!(Ycopy[thread]->UU[k] = malloc(F->list[k]->ldu*F->list[k]->ldu*sizeof(double)))) {
              fact_error = CHOMPACK_NOMEMORY;
              goto factorization_error;
            }

            if (update_UU_factor(Ycopy[thread], k, z, work + thread*3*F->n)) {
              FREE(Ycopy[thread]->UU[k]);
              fact_error = CHOMPACK_FACTORIZATION_ERR;
              goto factorization_error;
            }

            if (--dealloc_cnt[thread*F->nCliques + F->list[k]->anc->listidx] == 0)
              FREE(Ycopy[thread]->UU[F->list[k]->anc->listidx]);

            dtrmm_(&cL, &cL, &cT, &cN, &nU(F,k), &nS(F,k),
                &one, Ycopy[thread]->UU[k], &F->list[k]->ldu, dX[numfact]->US[k], &nU(F,k));

            dtrsm_(&cR, &cL, &cT, &cN, &nU(F,k), &nS(F,k), &one, R->SS[k], &nS(F,k),
                dX[numfact]->US[k], &nU(F,k));

            if (F->list[k]->nchildren == 0) FREE(Ycopy[thread]->UU[k]);
          }
        }
      } else { // inv=false, adj=true

        for (k=0; k<F->nCliques; k++)
          dealloc_cnt[thread*F->nCliques + k] = F->list[k]->nchildren;

        for (k=0; k<F->nCliques; k++) {

          /* dX_{Sk,Sk} := P_{k,k}^{-T} * dX_{Sk,Sk} * P_{k,k}^{-1},
                D_{Sk,Sk} = P_{k,k}*P_{k,k}^T */
          unpack(dX[numfact]->SS[k], nS(F,k));

          dtrsm_(&cL, &cL, &cT, &cN,
              &nS(F,k), &nS(F,k), &one, R->SS[k], &nS(F,k),
              dX[numfact]->SS[k], &nS(F,k));

          dtrsm_(&cR, &cL, &cN, &cN,
              &nS(F,k), &nS(F,k), &one, R->SS[k], &nS(F,k),
              dX[numfact]->SS[k], &nS(F,k));

          /*  dX_{Uk,Sk} = Q_k * dX_{Uk,Sk} * P_k^{-1},  Yuu = Q * Q^T */

          if (nU(F,k)) {
            if (!(Ycopy[thread]->UU[k] = malloc(F->list[k]->ldu*F->list[k]->ldu*sizeof(double)))) {
              fact_error = CHOMPACK_NOMEMORY;
              goto factorization_error;
            }

            if (update_UU_factor(Ycopy[thread], k, z, work + thread*3*F->n)) {
              FREE(Ycopy[thread]->UU[k]);
              fact_error = CHOMPACK_FACTORIZATION_ERR;
              goto factorization_error;
            }

            if (--dealloc_cnt[thread*F->nCliques + F->list[k]->anc->listidx] == 0) {
              FREE(Ycopy[thread]->UU[F->list[k]->anc->listidx]);
            }

            dtrmm_(&cL, &cL, &cN, &cN, &nU(F,k), &nS(F,k),
                &one, Ycopy[thread]->UU[k], &F->list[k]->ldu, dX[numfact]->US[k], &nU(F,k));

            dtrsm_(&cR, &cL, &cN, &cN, &nU(F,k), &nS(F,k), &one, R->SS[k], &nS(F,k),
                dX[numfact]->US[k], &nU(F,k));

            if (F->list[k]->nchildren == 0) FREE(Ycopy[thread]->UU[k]);
          }
        }

        for (k=0; k<F->nCliques; k++)
          dealloc_cnt[thread*F->nCliques + k] = F->list[k]->nchildren;

        for (k=0; k<F->nCliques; k++) {

          if (nU(F,k)) {

            double zero = 0.0;
            int n = nU(F,k)*nS(F,k), inc = 1;

            /*  dX_{Sk,Sk} := dX_{Sk,Sk} + R_{Uk,Sk}^T*dX_{Uk,Uk}*R_{Uk,Sk}
                   -(R_{Uk,Sk}^T*dX_{Uk,Sk} + dX_{Uk,Sk}^T*R_{Uk,Sk})          */

            /* dX_{Uk,Sk} := dX_{Uk,Sk} - dX_{Uk,Uk}*R_{Uk,Sk} */

            dsyr2k_(&cL, &cT, &nS(F,k), &nU(F,k), &minusone, R->US[k], &nU(F,k),
                dX[numfact]->US[k], &nU(F,k), &one, dX[numfact]->SS[k], &nS(F,k));

            dX[numfact]->UU[k] = malloc(nU(F,k)*nU(F,k)*sizeof(double));

            cliquecopy(dX[numfact], F->list[k], 0);

            if (--dealloc_cnt[thread*F->nCliques + F->list[k]->anc->listidx] == 0) {
              FREE(dX[numfact]->UU[F->list[k]->anc->listidx]);
            }

            dsymm_(&cL, &cL, &nU(F,k), &nS(F,k), &one, dX[numfact]->UU[k], &nU(F,k),
                R->US[k], &nU(F,k), &zero, tUS + thread*maxUS, &nU(F,k));

            dgemm_(&cT, &cN, &nS(F,k), &nS(F,k), &nU(F,k), &one, R->US[k], &nU(F,k),
                tUS + thread*maxUS, &nU(F,k), &one, dX[numfact]->SS[k], &nS(F,k));

            daxpy_(&n, &minusone, tUS + thread*maxUS, &inc, dX[numfact]->US[k], &inc);

            if (F->list[k]->nchildren == 0) FREE(dX[numfact]->UU[k]);
          }
        }
      }
    } //     if (!inv) { ... } else { ... }
    else {

      /* Factor solution to Newton system */
      if (adj) {

        int k;
        for (k=0; k<F->nCliques; k++) {

          if (nU(F,k)) {

            dX[numfact]->UU[k] = malloc(nU(F,k)*nU(F,k)*sizeof(double));
            cliquecopy(dX[numfact], F->list[k], 0);

          }
        }

        for (k=F->nCliques-1; k>=0; k--) {

          if (nU(F,k)) {

            double zero = 0.0;
            int n = nU(F,k)*nS(F,k), inc = 1;

            /*  dX_{Sk,Sk} := dX_{Sk,Sk} + R_{Uk,Sk}^T*dX_{Uk,Uk}*R_{Uk,Sk}
                     +(R_{Uk,Sk}^T*dX_{Uk,Sk} + dX_{Uk,Sk}^T*R_{Uk,Sk})       */

            /* dX_{Uk,Sk} := dX_{Uk,Sk} + dX_{Uk,Uk}*R_{Uk,Sk} */

            dsyr2k_(&cL, &cT, &nS(F,k), &nU(F,k), &one, R->US[k], &nU(F,k),
                dX[numfact]->US[k], &nU(F,k), &one, dX[numfact]->SS[k], &nS(F,k));

            dsymm_(&cL, &cL, &nU(F,k), &nS(F,k), &one, dX[numfact]->UU[k], &nU(F,k),
                R->US[k], &nU(F,k), &zero, tUS + thread*maxUS, &nU(F,k));

            dgemm_(&cT, &cN, &nS(F,k), &nS(F,k), &nU(F,k), &one, R->US[k], &nU(F,k),
                tUS + thread*maxUS, &nU(F,k), &one, dX[numfact]->SS[k], &nS(F,k));

            daxpy_(&n, &one, tUS + thread*maxUS, &inc, dX[numfact]->US[k], &inc);
          }

          FREE(dX[numfact]->UU[k]);
        }

        for (k=0; k<F->nCliques; k++)
          dealloc_cnt[thread*F->nCliques + k] = F->list[k]->nchildren;

        for (k=0; k<F->nCliques; k++) {

          /* dX_{Sk,Sk} := P_{k,k}^{T} * dX_{Sk,Sk} * P_{k,k},
                D_{Sk,Sk} = P_{k,k}*P_{k,k}^T */
          unpack(dX[numfact]->SS[k], nS(F,k));

          dtrmm_(&cL, &cL, &cT, &cN,
              &nS(F,k), &nS(F,k), &one, R->SS[k], &nS(F,k),
              dX[numfact]->SS[k], &nS(F,k));

          dtrmm_(&cR, &cL, &cN, &cN,
              &nS(F,k), &nS(F,k), &one, R->SS[k], &nS(F,k),
              dX[numfact]->SS[k], &nS(F,k));

          /*  dX_{Uk,Sk} = Q_k^{-1} * dX_{Uk,Sk} * P_k,  Yuu = Q * Q^T */
          if (nU(F,k)) {

            if (!(Ycopy[thread]->UU[k] = malloc(F->list[k]->ldu*F->list[k]->ldu*sizeof(double)))) {
              fact_error = CHOMPACK_NOMEMORY;
              goto factorization_error;
            }

            if (update_UU_factor(Ycopy[thread], k, z, work + thread*3*F->n)) {
              FREE(Ycopy[thread]->UU[k]);
              fact_error = CHOMPACK_FACTORIZATION_ERR;
              goto factorization_error;
            }

            if (--dealloc_cnt[thread*F->nCliques + F->list[k]->anc->listidx] == 0) {
              FREE(Ycopy[thread]->UU[F->list[k]->anc->listidx]);
            }

            dtrsm_(&cL, &cL, &cN, &cN, &nU(F,k), &nS(F,k),
                &one, Ycopy[thread]->UU[k], &F->list[k]->ldu, dX[numfact]->US[k], &nU(F,k));

            dtrmm_(&cR, &cL, &cN, &cN, &nU(F,k), &nS(F,k), &one, R->SS[k], &nS(F,k),
                dX[numfact]->US[k], &nU(F,k));

            if (F->list[k]->nchildren == 0) FREE(Ycopy[thread]->UU[k]);
          }
        }
      } else {

        /* Adj = false */

        for (k=0; k<F->nCliques; k++)
            dealloc_cnt[thread*F->nCliques + k] = F->list[k]->nchildren;

        for (k=0; k<F->nCliques; k++) {

          /* dX_{Sk,Sk} := P_{k,k} * dX_{Sk,Sk} * P_{k,k}^{T},
                D_{Sk,Sk} = P_{k,k}*P_{k,k}^T */
          unpack(dX[numfact]->SS[k], nS(F,k));

          dtrmm_(&cL, &cL, &cN, &cN, &nS(F,k), &nS(F,k), &one, R->SS[k], &nS(F,k),
              dX[numfact]->SS[k], &nS(F,k));

          dtrmm_(&cR, &cL, &cT, &cN, &nS(F,k), &nS(F,k), &one, R->SS[k], &nS(F,k),
              dX[numfact]->SS[k], &nS(F,k));

          /*  dX_{Uk,Sk} = Q_k^{-T} * dX_{Uk,Sk} * P_k^T,  Yuu = Q * Q^T */
          if (nU(F,k)) {

            if (!(Ycopy[thread]->UU[k] = malloc(F->list[k]->ldu*F->list[k]->ldu*sizeof(double)))) {
              fact_error = CHOMPACK_NOMEMORY;
              goto factorization_error;
            }

            if (update_UU_factor(Ycopy[thread], k, z, work + thread*3*F->n)) {
              FREE(Ycopy[thread]->UU[k]);
              fact_error = CHOMPACK_FACTORIZATION_ERR;
              goto factorization_error;
            }

            if (--dealloc_cnt[thread*F->nCliques + F->list[k]->anc->listidx] == 0) {
              FREE(Ycopy[thread]->UU[F->list[k]->anc->listidx]);
            }

            dtrsm_(&cL, &cL, &cT, &cN, &nU(F,k), &nS(F,k),
                &one, Ycopy[thread]->UU[k], &F->list[k]->ldu, dX[numfact]->US[k], &nU(F,k));

            dtrmm_(&cR, &cL, &cT, &cN, &nU(F,k), &nS(F,k), &one, R->SS[k], &nS(F,k),
                dX[numfact]->US[k], &nU(F,k));

            if (F->list[k]->nchildren == 0) FREE(Ycopy[thread]->UU[k]);
          }
        }

        for (k=0; k<F->nCliques; k++) {

          if (nU(F,k)) {

            /*  dX_{Uk,Uk} := dX_{Uk,Uk} + R_{Uk,Sk}*dX_{Sk,Sk}*R_{Uk,Sk}'
                   +(R_{Uk,Sk}*dX_{Uk,Sk}' + dX_{Uk,Sk}*R_{Uk,Sk}')          */

            /* dX_{Uk,Sk} := dX_{Uk,Sk} + R_{Uk,Sk}*dX_{Sk,Sk} */

            double zero = 0.0;
            int n = nU(F,k)*nS(F,k), inc = 1;

            double *tUU = malloc(nU(F,k)*nU(F,k)*sizeof(double));
            if (!tUU) {
              fact_error = CHOMPACK_NOMEMORY;
              goto factorization_error;
            }

            dsyr2k_(&cL, &cN, &nU(F,k), &nS(F,k), &one, R->US[k], &nU(F,k),
                dX[numfact]->US[k], &nU(F,k), &zero, tUU, &nU(F,k));

            dsymm_(&cR, &cL, &nU(F,k), &nS(F,k), &one, dX[numfact]->SS[k],
                &nS(F,k), R->US[k], &nU(F,k), &zero, tUS + thread*maxUS, &nU(F,k));

            dgemm_(&cN, &cT, &nU(F,k), &nU(F,k), &nS(F,k), &one, tUS + thread*maxUS,
                &nU(F,k), R->US[k], &nU(F,k), &one, tUU, &nU(F,k));

            daxpy_(&n, &one, tUS + thread*maxUS, &inc, dX[numfact]->US[k], &inc);

            cliquecopy_nmf(dX[numfact], F->list[k], 1, tUU);

            free(tUU);
          }
        }

      }
    }
    factorization_error:
    ;
  } // for (numfact = 0; ...

  free(dealloc_cnt);
  free(work);
  free(tUS);

  for (k=0; k<nthreads-1; k++)
    chordalmatrix_destroy(Ycopy[k]);
  free(Ycopy);

  return fact_error;
}

double dot(const chordalmatrix *X, const chordalmatrix *Y)
{
  int i, j, k;
  double v = 0.0;

  cliqueforest *F = X->F;
  for (k=0; k<F->nCliques; k++) {

    for (j=0; j<nS(F,k); j++) {
      for (i=0; i<nU(F,k); i++)
        v += 2*X->US[k][i+j*nU(F,k)]*Y->US[k][i+j*nU(F,k)];

      for (i=j+1; i<nS(F,k); i++)
        v += 2*X->SS[k][i+j*nS(F,k)]*Y->SS[k][i+j*nS(F,k)];

      v += X->SS[k][j+j*nS(F,k)]*Y->SS[k][j+j*nS(F,k)];
    }
  }
  return v;
}

void axpy(const chordalmatrix *X, chordalmatrix *Y, double a) {

  int i, j, k;

  cliqueforest *F = X->F;
  for (k=0; k<F->nCliques; k++) {

    for (j=0; j<nS(F,k); j++) {
      for (i=0; i<nU(F,k); i++)
        Y->US[k][i+j*nU(F,k)] = a*X->US[k][i+j*nU(F,k)] + Y->US[k][i+j*nU(F,k)];

      for (i=j; i<nS(F,k); i++)
        Y->SS[k][i+j*nS(F,k)] = a*X->SS[k][i+j*nS(F,k)] + Y->SS[k][i+j*nS(F,k)];
    }
  }
}

void scal(double a, chordalmatrix *X) {

  int i, j, k;

  cliqueforest *F = X->F;
  for (k=0; k<F->nCliques; k++) {

    for (j=0; j<nS(F,k); j++) {
      for (i=0; i<nU(F,k); i++)
        X->US[k][i+j*nU(F,k)] *= a;

      for (i=j; i<nS(F,k); i++)
        X->SS[k][i+j*nS(F,k)] *= a;
    }
  }
}

/*
 * Solution of triangular set of equations.
 *
 * R contains the Cholesky factor P * R * R' * P' of a
 * positive definite matrix.
 *
 * On exit, X is overwritten with
 * X := P * R * X        if sys = 0
 * X := R' * P' * X      if sys = 1
 * X := R^{-1} * P * X   if sys = 2
 * X := P' * R^{-T} * X  if sys = 3
 */
int solve(const chordalmatrix *R, double *X, int n, int mode) {

  int k;
  char cN = 'N', cT = 'T', cL = 'L';
  double dzero = 0.0, dminusone = -1.0, done = 1.0;

  cliqueforest *F = R->F;
  chordalvec *y;

  if (mode == 0) {   /* X := R^{-1} * P' * X */

    int l;
    y = dense_to_chordalvec(F, X, n, F->p);
    if (!y) return CHOMPACK_NOMEMORY;

    for (k=F->nCliques-1; k>=0; k--) {

      if (nU(F,k)) {
        if (!(y->U[k] = calloc(nU(F,k)*n,sizeof(double)))) {
          chordalvec_destroy(y);
          return CHOMPACK_NOMEMORY;
        }
      }

      for (l=0; l<F->list[k]->nchildren; l++)
        {
          int chidx = F->list[k]->children[l]->listidx;
          cliquecopy_vec(y, F->list[chidx], 1);
          FREE(y->U[chidx]);
        }

      if (nU(F,k)) {

        dgemm_(&cN, &cN, &nU(F,k), &n, &nS(F,k),
            &dminusone, R->US[k], &nU(F,k), y->S[k], &nS(F,k),
            &done, y->U[k], &nU(F,k));

      }

      dtrsm_(&cL, &cL, &cN, &cN,
          &nS(F,k), &n, &done, R->SS[k], &nS(F,k), y->S[k], &nS(F,k));
    }

    chordalvec_to_dense(y, X, NULL);
    chordalvec_destroy(y);
  }

  else if (mode == 1) { /* X := P * R^{-T} * X */

    int *dealloc_cnt = malloc(F->nCliques*sizeof(int));
    if (!dealloc_cnt) return CHOMPACK_NOMEMORY;

    y = dense_to_chordalvec(F, X, n, NULL);
    if (!y) { free(dealloc_cnt); return CHOMPACK_NOMEMORY; }

    for (k=0; k<F->nCliques; k++)
      dealloc_cnt[k] = F->list[k]->nchildren;

    for (k=0; k<F->nCliques; k++) {

      dtrsm_(&cL, &cL, &cT, &cN,
          &nS(F,k), &n, &done, R->SS[k], &nS(F,k), y->S[k], &nS(F,k));

      if (nU(F,k)) {

        if (!(y->U[k] = malloc(nU(F,k)*n*sizeof(double)))) {
          chordalvec_destroy(y);
          return CHOMPACK_NOMEMORY;
        }

        cliquecopy_vec(y, F->list[k], 0);

        if (--dealloc_cnt[F->list[k]->anc->listidx] == 0) {
          FREE(y->U[F->list[k]->anc->listidx]);
        }

        dgemm_(&cT, &cN, &nS(F,k), &n, &nU(F,k),
            &dminusone, R->US[k], &nU(F,k), y->U[k], &nU(F,k),
            &done, y->S[k], &nS(F,k));

        if (F->list[k]->nchildren == 0) FREE(y->U[k]);
      }
    }

    free(dealloc_cnt);

    chordalvec_to_dense(y, X, F->p);
    chordalvec_destroy(y);
  }
  else if (mode > 1) {

    double *t;
    int maxU = 0;
    for (k=0; k<F->nCliques; k++)
      maxU = MAX(maxU, nU(F,k));

    if (!(t = calloc(maxU*n,sizeof(double))))
      return CHOMPACK_NOMEMORY;

    if (mode == 2) { /* X := P * R * X */

      y = dense_to_chordalvec(F, X, n, NULL);
      if (!y) {
        free(t);
        return CHOMPACK_NOMEMORY;
      }

      for (k=0; k<F->nCliques; k++) {

        /* x_{Sk} := L*x_{Sk} */
        dtrmm_(&cL, &cL, &cN, &cN,
            &nS(F,k), &n, &done, R->SS[k], &nS(F,k), y->S[k], &nS(F,k));

        if (nU(F,k)) {

          /* x_{Uk} := x_{Uk} + R_{Uk,Sk}*x_{Sk} */
          dgemm_(&cN, &cN, &nU(F,k), &n, &nS(F,k),
              &done, R->US[k], &nU(F,k), y->S[k], &nS(F,k),
              &dzero, t, &nU(F,k));

          cliquecopy_vec_nmf(y, F->list[k], 1, t);
        }
      }

      chordalvec_to_dense(y, X, F->p);
      chordalvec_destroy(y);
    }
    else { /* X := R' * P' * X */

      y = dense_to_chordalvec(F, X, n, F->p);
      if (!y) {
        free(t);
        return CHOMPACK_NOMEMORY;
      }

      for (k=F->nCliques-1; k>=0; k--) {

        if (nU(F,k)) {

          cliquecopy_vec_nmf(y, F->list[k], 0, t);

          dgemm_(&cT, &cN, &nS(F,k), &n, &nU(F,k),
              &done, R->US[k], &nU(F,k), t, &nU(F,k),
              &done, y->S[k], &nS(F,k));

        }

        dtrmm_(&cL, &cL, &cT, &cN,
            &nS(F,k), &n, &done, R->SS[k], &nS(F,k), y->S[k], &nS(F,k));
      }

      chordalvec_to_dense(y, X, NULL);
      chordalvec_destroy(y);

    }
    free(t);
  }

  return CHOMPACK_OK;
}

double logdet(const chordalmatrix *X) {

  int i, k;
  double ret = 0.0;

  cliqueforest *F = X->F;

  for (k=0; k<F->nCliques; k++) {

    for (i=0; i<nS(F,k); i++) {

      ret += log( X->SS[k][i + i*nS(F,k)] );

    }
  }

  return ret;
}

void syr1(chordalmatrix *X, double *y, double a, double b) {

  cliqueforest *F = X->F;

  int i,j,k,o;
  for (k=0; k<F->nCliques; k++) {

    double yj;
    int *Sk = F->list[k]->S;
    int *Uk = F->list[k]->U;

    for (j=0; j<nS(F,k); j++) {

      yj = y[ F->p[ Sk[j]] ];

      o=j*nS(F,k);
      for (i=j; i<nS(F,k); i++) {
        X->SS[k][i + o] *= b;
        X->SS[k][i + o] += a*y[ F->p[ Sk[i] ] ]*yj;
      }

      o = j*nU(F,k);
      for (i=0; i<nU(F,k); i++) {
        X->US[k][i + o] *= b;
        X->US[k][i + o] += a*y[ F->p[ Uk[i] ] ]*yj;
      }
    }
  }
}

void syr2(chordalmatrix *X, double *y, double *z, double a, double b) {

  cliqueforest *F = X->F;

  int i,j,k,o;
  for (k=0; k<F->nCliques; k++) {

    double yj, zj;
    int *Sk = F->list[k]->S;
    int *Uk = F->list[k]->U;

    for (j=0; j<nS(F,k); j++) {

      yj = y[ F->p[ Sk[j]] ];
      zj = z[ F->p[ Sk[j]] ];

      o=j*nS(F,k);
      for (i=j; i<nS(F,k); i++) {
        X->SS[k][i + o] *= b;
        X->SS[k][i + o] += a*y[ F->p[ Sk[i] ] ]*zj +
            a*z[ F->p[ Sk[i] ] ]*yj;
      }

      o = j*nU(F,k);
      for (i=0; i<nU(F,k); i++) {
        X->US[k][i + o] *= b;
        X->US[k][i + o] += a*y[ F->p[ Uk[i] ] ]*zj +
            a*z[ F->p[ Uk[i] ] ]*yj;
      }
    }
  }
}
