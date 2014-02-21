#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "chompack.h"

clique ** cliquelist_create(const cliqueforest *F);

extern void dtrmm_(char *side, char *uplo, char *transa, char *diag,
    int *m, int *n, double *alpha, double *A, int *lda, double *B,
    int *ldb);

#ifdef DEBUG
static void clique_print(clique *K) {

  int i;
  printf("CLIQUE %i\n", K->listidx);
  printf("S: ");
  for (i=0; i<K->ns; i++)
    printf("%i ",K->S[i]);

  printf("\nU: ");
  for (i=0; i<K->nu; i++)
    printf("%i ",K->U[i]);

  if (K->anc)
    printf("\nanc: %i\n", K->anc->listidx);
}

static void cliquetree_print(clique *K) {

  int i;
  for (i=0; i<K->nchildren; i++)
    cliquetree_print(K->children[i]);

  clique_print(K);
}
#endif

static void clique_destroy(clique *K) {

  int i = 0;
  for (i=0; i<K->nchildren; i++)
    clique_destroy(K->children[i]);

  free(K->Uidx);
  free(K->U);
  free(K->S);
  free(K->Uanc);
  free(K->children);
  free(K);
}

void cliqueforest_destroy(cliqueforest *F) {

  int i;
  for (i=0; i<F->nCliques; i++) {

    free(F->list[i]->U);
    free(F->list[i]->S);
    free(F->list[i]->Uidx);
    free(F->list[i]->Uanc);
    free(F->list[i]->children);
  }

  free(F->K);
  free(F->roots);
  free(F->list);
  free(F->p); free(F->ip);
  free(F);
}

int cliqueforest_create(cliqueforest **F, const ccs *X, const int_t *p) {

  int n = X->nrows;
  int_t *colptr = X->colptr, *rowind = X->rowind;

  int_t k, j, cliqueSize;
  int *size_madj = NULL, *cliqueCnt = NULL, *inNew = NULL, *nNew = NULL;
  int *anc = NULL, *essNodes = NULL, *nchildren = NULL;
  clique *K = NULL;
  int nCliques = 0;   /* # cliques */
  int nroots = 0;

  int_t *p_  = malloc(n*sizeof(int_t));
  int_t *ip_ = malloc(n*sizeof(int_t));
  if (!p_ || !ip_) {
    free(p_); free(ip_); return CHOMPACK_NOMEMORY;
  }

  for (k=0; k<n; k++) {
    p_[k]  = (p ? p[k] : k); ip_[p_[k]] = k;
  }

  if (!(size_madj = calloc(n, sizeof(int)))) goto err_cleanup;
  for (j=0; j<n; j++) {
    for (k=colptr[j]; k<colptr[j+1]; k++) {
      if (rowind[k] == j) {
        size_madj[j] = colptr[j+1]-k-1;
        break;
      }
    }
  }

  /* # cliques containing node k */
  if (!(cliqueCnt = calloc(n,sizeof(int)))) goto err_cleanup;

  /* node k belongs to new(k) */
  if (!(inNew = malloc(n*sizeof(int)))) goto err_cleanup;
  for (k=0; k<n; k++) inNew[k] = -1;

  /* # nodes in new(k) */
  if (!(nNew = calloc(n,sizeof(int)))) goto err_cleanup;

  /* anc(k) is the ancestor to clique k */
  if (!(anc = malloc(n*sizeof(int)))) goto err_cleanup;
  for (k=0; k<n; k++) anc[k] = -1;

  if (!(essNodes = malloc(n*sizeof(int)))) goto err_cleanup;

  for (k=0; k<n; k++) {

    if (inNew[k] == -1 ) {  /* node k is an essential node */

      nCliques++;
      essNodes[nCliques-1] = k;
      cliqueSize = size_madj[k];
      inNew[k] = nCliques-1;
      cliqueCnt[k]++;
      nNew[nCliques-1] = 1;
      anc[nCliques-1] = -1;

      for (j=colptr[k+1]-size_madj[k]; j<colptr[k+1]; j++) {

        int w = rowind[j];
        cliqueCnt[w]++;

        if (anc[nCliques-1] == -1) {
          cliqueSize--;
          if ((cliqueSize == size_madj[w]) && (inNew[w] == -1)) {
            nNew[nCliques-1]++;
            inNew[w] = nCliques-1;


          } else {
            anc[nCliques-1]= w;
          }
        }
      }
    }
  }

  if (!(nchildren = calloc(nCliques,sizeof(int)))) goto err_cleanup;

  if (!(K = malloc(nCliques*sizeof(clique)))) goto err_cleanup;

  for (k=0; k<nCliques; k++) {

    int v = essNodes[k];
    K[k].ns = nNew[k];
    K[k].nu = size_madj[v]+1-nNew[k];
    K[k].S = malloc(K[k].ns*sizeof(int));
    K[k].U = malloc(K[k].nu*sizeof(int));

    if (!K[k].S || !K[k].U) {
      int l;
      for (l=0; l<=k; l++) {
        free(K[l].S);
        free(K[l].U);
        goto err_cleanup;
      }
    }

    for (j=0; j<K[k].ns; j++)
      K[k].S[j] = rowind[colptr[v+1]-size_madj[v]-1+j];

    for (j=0; j<K[k].nu; j++)
      K[k].U[j] = rowind[colptr[v+1]-size_madj[v]-1+K[k].ns+j];

    if (K[k].nu == 0) nroots++;
    if (anc[k] != -1) {
      K[k].anc = &K[ inNew[anc[k]] ];
      nchildren[ inNew[anc[k]] ]++;
    } else {
      K[k].anc = 0;
    }

    if (!(K[k].Uanc = malloc(K[k].nu*sizeof(clique *)))) {
      int l;
      for (l=0; l<=k; l++) {
        free(K[l].S); free(K[l].U);
        goto err_cleanup;
      }
    }

    for (j=0; j<K[k].nu; j++)
      K[k].Uanc[j] = &K[ inNew[K[k].U[j]] ];

    K[k].nchildren = 0;

  }

  /*
   * Clique-merging can be done here
   */

  for (k=0; k<nCliques; k++) {

    if (!(K[k].children = malloc(nchildren[k]*sizeof(clique *)))) {
      int l;
      for (l=0; l<nCliques; l++) {
        free(K[l].S); free(K[l].U); free(K[l].Uanc);
        goto err_cleanup;
      }
    }
  }

  *F = malloc(sizeof(cliqueforest));
  if (!(*F)) {
    clique_destroy(K);
    goto err_cleanup;
  }

  if (!((*F)->roots = malloc(nroots*sizeof(clique *)))) {
    clique_destroy(K);
    free(F);
    goto err_cleanup;
  }

  (*F)->nRoots = 0;
  (*F)->nCliques = nCliques;
  (*F)->n = n;

  for (k=0; k<nCliques; k++)
    {
      if (K[k].anc) {
        K[k].anc->children[ K[k].anc->nchildren++ ] = &K[k];
        K[k].ldu = MAX(K[k].anc->nu, K[k].nu);
      }
      if (K[k].nu == 0) (*F)->roots[(*F)->nRoots++] = &K[k];

     if (!K[k].anc) K[k].ldu = K[k].nu;
    }

  if (!((*F)->list = cliquelist_create(*F))) {
    clique_destroy(K);
    free(*F);
    goto err_cleanup;
  }

  for (k=0; k<nCliques; k++) {

    int j = 0, l = 0;
    (*F)->list[k]->listidx = k;

    if (!(K[k].Uidx = malloc(K[k].nu*sizeof(int)))) {
      for (j=0; j<k; j++)
        free(K[j].Uidx);

      clique_destroy(K);
      free(*F);
      goto err_cleanup;
    }

    while (j < K[k].nu && l < K[k].anc->ns) {
      if (K[k].anc->S[l] != K[k].U[j])
        l++;
      else
        K[k].Uidx[j++] = l++;
    }
    l = 0;
    K[k].Uidx_ns = j;
    while (j < K[k].nu && l < K[k].anc->nu) {
      if (K[k].anc->U[l] != K[k].U[j])
        l++;
      else
        K[k].Uidx[j++] = l++;
    }
  }

  (*F)->p = p_; (*F)->ip = ip_; (*F)->K = K;

  (*F)->allocsize = 0;
  for (k=0; k<nCliques; k++)
    (*F)->allocsize += (K[k].nu+K[k].ns)*K[k].ns;

  free(size_madj);
  free(cliqueCnt);
  free(nNew);
  free(inNew);
  free(anc);
  free(essNodes);
  free(nchildren);
  return CHOMPACK_OK;

  err_cleanup:

  free(p_); free(ip_);
  free(size_madj);
  free(cliqueCnt);
  free(nNew);
  free(inNew);
  free(anc);
  free(essNodes);
  free(nchildren);
  free(K);
  return CHOMPACK_NOMEMORY;
}

/* Stores the cliques in a list in reverse topological ordering */
int cliquelist(clique *K, clique **list) {

  int i, n = 0;
  for (i=0; i<K->nchildren; i++) {
    n += cliquelist(K->children[i], &list[n+1]);
  }

  list[0] = K;

  return n+1;
}

clique ** cliquelist_create(const cliqueforest *F) {

  int i, n = 0;
  clique **L = malloc(F->nCliques*sizeof(void *));
  if (!L) return NULL;

  for (i=0; i<F->nRoots; i++)
    n += cliquelist(F->roots[i], &L[n]);

  return L;
}

void chordalmatrix_destroy(chordalmatrix *X)
{
  int k;
  for (k=0; k<X->F->nCliques; k++) {
    FREE(X->UU[k]);
  }

  free(X->US); free(X->SS); free(X->UU); free(X->data); free(X);
}

/*
 * Add child clique X_{Uc,Uc} to parent X_{Vk,Vk}
 */
void cliquecopy(chordalmatrix *X, const clique *child, int toparent) {

  int i, j;

  clique *par = child->anc;
  int *uidx = child->Uidx;
  double *SSp = X->SS[par->listidx];
  double *USp = X->US[par->listidx];
  double *UUp = X->UU[par->listidx];
  double *UUc = X->UU[child->listidx];

  int k1, k2;
  for (j=0; j<child->Uidx_ns; j++) {
    k1 = uidx[j]*par->ns;
    k2 = j*child->nu;
    for (i=j; i<child->Uidx_ns; i++) {
      if (toparent)
        SSp[uidx[i] + k1] += UUc[i + k2];
      else
        UUc[i + k2] = SSp[uidx[i] + k1];
    }

    k1 = uidx[j]*par->nu;
    for (i=child->Uidx_ns; i<child->nu; i++) {
      if (toparent)
        USp[uidx[i] + k1] += UUc[i + k2];
      else
        UUc[i + k2] = USp[uidx[i] + k1];
    }
  }

  for (j=child->Uidx_ns; j<child->nu; j++) {
    k1 = uidx[j]*par->nu;
    k2 = j*child->nu;
    for (i=j; i<child->nu; i++) {
      if (toparent)
        UUp[uidx[i] + k1] += UUc[i + k2];
      else
        UUc[i + k2] = UUp[uidx[i] + k1];
    }
  }
}

/*
 * Copy data between child clique X_{Uc,Uc} and parent X_{Vk,Vk}
 * Used by non-multifrontal routines
 */
void cliquecopy_nmf(chordalmatrix *X, const clique *child, int toparent, double *src) {

  int j, k, l, p = 0;
  double *x = (src ? src : X->UU[child->listidx]);

  for (j=0; j<child->nu; j++) {

    int k1, k2;
    clique *par = child->Uanc[j];

    p = 0;
    while (par->S[p] != child->U[j]) p++;

    k = j; l = p;
    k1 = p*par->ns;
    k2 = j*child->nu;
    while ((l < par->ns) && (k < child->nu)) {

      if (child->U[k] == par->S[l]) {
        if (toparent)
          X->SS[par->listidx][l + k1] += x[(k++) + k2];
        else
          x[(k++) + k2] = X->SS[par->listidx][l + k1];
      }
      l++;
    }

    l = 0;
    k1 = p*par->nu;
    while ((l < par->nu) && (k < child->nu)) {
      if (child->U[k] == par->U[l]) {
        if (toparent)
          X->US[par->listidx][l + k1] += x[(k++) + k2];
        else
          x[(k++) + k2] = X->US[par->listidx][l + k1];
      }
      l++;
    }
  }
}

/*
 * Copy data between child clique X_{Uc} and parent clique X_{Vk} in vector routines
 */
void cliquecopy_vec(chordalvec *X, const clique *child, int toparent) {

  int i, j;

  int *uidx = child->Uidx;
  clique *par = child->anc;

  double *Sp = X->S[par->listidx];
  double *Up = X->U[par->listidx];
  double *Uc = X->U[child->listidx];

  for (j=0; j<X->ncols; j++) {

    int k1 = j*par->ns, k2 = j*child->nu;
    for (i=0; i<child->Uidx_ns; i++) {
      if (toparent)
        Sp[uidx[i] + k1] += Uc[i + k2];
      else
        Uc[i + k2] = Sp[uidx[i] + k1];
    }

    k1 = j*par->nu;
    for (i=child->Uidx_ns; i<child->nu; i++) {
      if (toparent)
        Up[uidx[i] + k1] += Uc[i + k2];
      else
        Uc[i + k2] = Up[uidx[i] + k1];
    }
  }
}

/*
 * Copy data between child clique X_{Uc} to parent X_{Vk} in vector routines
 * Used by non-multifrontal versions
 */
void cliquecopy_vec_nmf(chordalvec *X, const clique *child, int toparent, double *src) {

  int j, k, p = 0;
  double *x = (src ? src : X->U[child->listidx]);

  k = 0;
  while (k<child->nu) {

    clique *par = child->Uanc[k];

    p = 0;
    while (par->S[p] != child->U[k]) p++;

    while ((p < par->ns) && (k < child->nu)) {

      if (child->U[k] == par->S[p]) {
        for (j=0; j<X->ncols; j++) {
          if (toparent)
            X->S[par->listidx][p + j*par->ns] += x[k + j*child->nu];
          else
            x[k + j*child->nu] = X->S[par->listidx][p + j*par->ns];
        }
        k++;
      }
      p++;
    }
  }
}

/* A must be reordered according to F->p before this call. */
chordalmatrix * ccs_to_chordalmatrix (const cliqueforest *F, const ccs *A) {

  int_t ci, cj, k, l;
  const ccs *Y = A;
  int allocsize;

  int_t *colptr = Y->colptr, *rowind = Y->rowind;
  double *val = Y->values;

  chordalmatrix *X = malloc(sizeof(chordalmatrix));
  if (!X) return NULL;

  X->US = malloc(F->nCliques*sizeof(void *));
  X->SS = malloc(F->nCliques*sizeof(void *));
  X->UU = malloc(F->nCliques*sizeof(void *));
  X->data = calloc(F->allocsize,sizeof(double));

  if (!X->US || !X->SS || !X->UU || !X->data) {
    free(X->US); free(X->SS); free(X->UU); free(X->data); free(X);
    return NULL;
  }

  X->F = (cliqueforest *)F;
  for (k=0, allocsize=0; k<F->nCliques; k++) {

    X->SS[k] = X->data + allocsize;
    X->US[k] = X->SS[k] + nS(F,k)*nS(F,k);
    allocsize += nS(F,k)*(nS(F,k) + nU(F,k));
    X->UU[k] = NULL;

    for (cj=0; cj<nS(F,k); cj++) {

      ci = 0;
      for (l = colptr[S(F,k)[cj]]; l < colptr[S(F,k)[cj]+1]; l++) {

        while ((ci < nU(F,k)) && (U(F,k)[ci] < rowind[l])) ci++;
        if (ci >= nU(F,k)) break;

        if (U(F,k)[ci] == rowind[l]) X->US[k][ci + cj*nU(F,k)] = val[l];
      }

      ci = 0;
      for (l = colptr[S(F,k)[cj]]; l < colptr[S(F,k)[cj]+1]; l++) {

        while ((ci < nS(F,k)) && (S(F,k)[ci] < rowind[l])) ci++;
        if (ci >= nS(F,k)) break;

        if (S(F,k)[ci] == rowind[l]) X->SS[k][ci + cj*nS(F,k)] = val[l];

      }
    }
  }

  return X;
}

chordalmatrix * dense_to_chordalmatrix(const cliqueforest *F, const double *x) {

  int i, j, k, allocsize;

  chordalmatrix *X = malloc(sizeof(chordalmatrix));
  if (!X) return NULL;

  X->UU = malloc(F->nCliques*sizeof(void *));
  X->US = malloc(F->nCliques*sizeof(void *));
  X->SS = malloc(F->nCliques*sizeof(void *));
  X->data = malloc(F->allocsize*sizeof(double));

  if (!X->UU || !X->US || !X->SS || !X->data) {
    free(X->US); free(X->SS); free(X->data); free(X);
    return NULL;
  }

  X->F = (cliqueforest *)F;
  allocsize = 0;
  for (k=0; k<F->nCliques; k++) {

    X->SS[k] = X->data + allocsize;
    X->US[k] = X->data + allocsize + nS(F,k)*nS(F,k);
    allocsize += nS(F,k)*(nS(F,k) + nU(F,k));
    X->UU[k] = NULL;

    for (j=0; j<nS(F,k); j++)
      for (i=j; i<nS(F,k); i++)
        X->SS[k][i + j*nS(F,k)] = x[ F->p[S(F,k)[i]] + F->p[S(F,k)[j]]*F->n ];

    for (j=0; j<nS(F,k); j++)
      for (i=0; i<nU(F,k); i++)
        X->US[k][i + j*nU(F,k)] = x[ F->p[U(F,k)[i]] + F->p[S(F,k)[j]]*F->n ];
  }

  return X;
}

ccs * chordalfactor_to_ccs (const chordalmatrix *A) {

  ccs *X;
  cliqueforest *F = A->F;
  int i, j, k;
  int_t *colptr = malloc( (F->n+1)*sizeof(int_t));
  if (!colptr) return NULL;

  colptr[0] = 0;
  for (k=0; k<F->nCliques; k++) {
    for (i=0; i<nS(F,k); i++) {
      colptr[S(F,k)[i]+1] = nS(F,k)-i + nU(F,k);
    }
  }

  for (i=0; i<F->n; i++) colptr[i+1] += colptr[i];

  if (!(X = alloc_ccs(F->n, F->n, colptr[F->n]))) {
    free(colptr);
    return NULL;
  }

  free(X->colptr);
  X->colptr = colptr;

  for (k=0; k<colptr[F->n]; k++)
    X->rowind[k] = -1;

  for (k=0; k<F->nCliques; k++) {

    double *T = malloc(nU(F,k)*nS(F,k)*sizeof(double));
    if (!T) {
      free_ccs(X);
      return NULL;
    }
    if (nU(F,k)) {

      char side = 'R', uplo = 'L', transA='N', diag='N';
      double alpha = 1.0;
      memcpy(T, A->US[k], nU(F,k)*nS(F,k)*sizeof(double));

      dtrmm_(&side, &uplo, &transA, &diag,
          &nU(F,k), &nS(F,k), &alpha, A->SS[k], &nS(F,k), T, &nU(F,k));
    }

    for (j=0; j<nS(F,k); j++) {
      for (i=j; i<nS(F,k); i++) {
        X->rowind[ colptr[S(F,k)[j]] +i-j ] = S(F,k)[i];
        ((double *)X->values)[ colptr[S(F,k)[j]] +i-j ] =
          A->SS[k][i + j*nS(F,k)];
      }

      for (i=0; i<nU(F,k); i++) {
        X->rowind[ colptr[S(F,k)[j]] + nS(F,k)-j + i] = U(F,k)[i];
        ((double *)X->values)[ colptr[S(F,k)[j]] + nS(F,k)-j + i ] = T[i + j*nU(F,k)];
      }
    }
    free(T);
  }

  return X;
}

ccs * chordalmatrix_to_ccs (const chordalmatrix *A) {

  int i, j, k;
  cliqueforest *F = A->F;
  ccs *X;
  int_t *colptr = malloc( (F->n+1)*sizeof(int_t));
  if (!colptr) return NULL;

  colptr[0] = 0;
  for (k=0; k<F->nCliques; k++) {
    for (i=0; i<nS(F,k); i++) {
      colptr[S(F,k)[i]+1] = nS(F,k)-i + nU(F,k);
    }
  }

  for (i=0; i<F->n; i++) colptr[i+1] += colptr[i];

  if (!(X = alloc_ccs(F->n, F->n, colptr[F->n]))) {
    free(colptr);
    return NULL;
  }

  free(X->colptr);
  X->colptr = colptr;

  for (k=0; k<colptr[F->n]; k++)
    X->rowind[k] = -1;

  for (k=0; k<F->nCliques; k++) {

    for (j=0; j<nS(F,k); j++) {
      for (i=j; i<nS(F,k); i++) {
        X->rowind[ colptr[S(F,k)[j]] +i-j ] = S(F,k)[i];
        ((double *)X->values)[ colptr[S(F,k)[j]] +i-j ] =
          A->SS[k][i + j*nS(F,k)];
      }

      for (i=0; i<nU(F,k); i++) {
        X->rowind[ colptr[S(F,k)[j]] + nS(F,k)-j + i] = U(F,k)[i];
        ((double *)X->values)[ colptr[S(F,k)[j]] + nS(F,k)-j + i ] =
          A->US[k][i + j*nU(F,k)];
      }
    }
  }
  return X;
}

chordalmatrix * chordalmatrix_copy(const chordalmatrix *A) {

  int k, allocsize;
  chordalmatrix *X = malloc(sizeof(chordalmatrix));
  if (!X) return NULL;

  X->US = malloc(A->F->nCliques*sizeof(void*));
  X->SS = malloc(A->F->nCliques*sizeof(void*));
  X->UU = malloc(A->F->nCliques*sizeof(void*));
  X->data = malloc(A->F->allocsize*sizeof(double));

  if (!X->US || !X->SS || !X->UU || !X->data) {
    free(X->US); free(X->SS); free(X->UU); free(X->data); free(X);
    return NULL;
  }

  cliqueforest *F = A->F;
  X->F = F;
  allocsize = 0;
  for (k=0; k<F->nCliques; k++) {

    X->SS[k] = X->data + allocsize;
    X->US[k] = X->data + allocsize + (nS(F,k)*nS(F,k));
    allocsize += nS(F,k)*(nS(F,k) + nU(F,k));
    X->UU[k] = NULL;
  }

  memcpy(X->data, A->data, F->allocsize*sizeof(double));
  X->F_py = A->F_py;
  return X;
}

chordalvec * alloc_chordalvec(const cliqueforest *F, int ncols) {

  int k, sizek;

  chordalvec *X = malloc(sizeof(chordalvec));
  if (!X) return NULL;

  X->U = malloc(F->nCliques*sizeof(void *));
  X->S = malloc(F->nCliques*sizeof(void *));
  X->data = malloc(F->n*ncols*sizeof(double));

  if (!X->U || !X->S || !X->data) {
    free(X->U); free(X->S); free(X->data); free(X);
    return NULL;
  }

  X->F = (cliqueforest *)F;
  X->ncols = ncols;
  for (k=0, sizek=0; k<F->nCliques; k++) {

    X->S[k] = X->data + sizek;
    X->U[k] = NULL;
    sizek += nS(F,k)*ncols;

  }
  return X;
}

void copy_dense_to_chordalvec(chordalvec *X, const double *x, const int_t *p, char trans) {

  int i, j, k, l;

  cliqueforest *F = X->F;
  for (k=0; k<F->nCliques; k++) {

    if (trans=='N') {

      for (j=0; j<X->ncols; j++)
        for (i=0, l=j*nS(F,k); i<nS(F,k); i++) {
          X->S[k][i + l] = x[(p ? p[S(F,k)[i]] : S(F,k)[i]) + j*F->n ];
        }
    } else {

      for (j=0; j<X->ncols; j++)
        for (i=0, l=j*nS(F,k); i<nS(F,k); i++) {
          X->S[k][i + l] = x[(p ? p[S(F,k)[i]] : S(F,k)[i])*X->ncols + j ];
        }

    }
  }
}

void sparse_to_chordalvec(chordalvec *X, int n, ccs *x, int coloffs, spa *s, int_t *p) {

  int i, j, k, l;

  cliqueforest *F = X->F;
  for (j=0; j<n; j++) {

    init_spa(s, x, coloffs+j);
    for (k=0; k<F->nCliques; k++) {

      if (p) {
        for (i=0, l=j*nS(F,k); i<nS(F,k); i++)
          X->S[k][i+l] = (s->nz[p[S(F,k)[i]]] ? s->val[p[S(F,k)[i]]] : 0.0);
      } else {
        for (i=0, l=j*nS(F,k); i<nS(F,k); i++)
          X->S[k][i+l] = (s->nz[S(F,k)[i]] ? s->val[S(F,k)[i]] : 0.0);
      }
    }
  }
}

chordalvec * dense_to_chordalvec(const cliqueforest *F, const double *x, int ncols, const int_t *p) {

  int i, j, k, sizek;

  chordalvec *X = malloc(sizeof(chordalvec));
  if (!X) return NULL;

  X->U = malloc(F->nCliques*sizeof(void *));
  X->S = malloc(F->nCliques*sizeof(void *));
  X->data = malloc(F->n*ncols*sizeof(double));

  if (!X->U || !X->S || !X->data) {
    free(X->U); free(X->S); free(X->data); free(X);
    return NULL;
  }

  X->F = (cliqueforest *)F;
  X->ncols = ncols;
  for (k=0, sizek=0; k<F->nCliques; k++) {

    int l;
    X->S[k] = X->data + sizek;
    X->U[k] = NULL;
    sizek += nS(F, k)*ncols;

    for (j=0; j<ncols; j++)
      for (i=0, l=j*nS(F,k); i<nS(F,k); i++) {
        X->S[k][i + l] = x[(p ? p[S(F,k)[i]] : S(F,k)[i]) + j*F->n ];
      }
  }
  return X;
}

void chordalvec_to_dense (const chordalvec *X, double *Y, const int_t *p) {

  cliqueforest *F = X->F;

  int i, j, k, l;

  for (k=0; k<F->nCliques; k++) {

    for (i=0; i<nS(F,k); i++) {
      for (j=0, l = p ? p[S(F,k)[i]] : S(F,k)[i]; j<X->ncols; j++) {

        Y[l + j*F->n] = X->S[k][i + j*nS(F,k)];
      }
    }
  }
}


void chordalvec_destroy(chordalvec *X)
{
  int k;
  for (k=0; k<X->F->nCliques; k++) {
    FREE(X->U[k]);
  }
  free(X->S); free(X->U); free(X->data); free(X);
}
