#ifndef __SPARSE__
#define __SPARSE__

#include "chtypes.h"

ccs * alloc_ccs(int_t nrows, int_t ncols, int nnz);
void free_ccs(ccs *obj);
ccs * perm(const ccs *A, int_t *p);
int iperm(int n, int_t *order, int_t *iorder);
ccs * chordalembedding(const ccs *X, int *nfill);
ccs * transpose(const ccs *A, int_t *p);
ccs * symmetrize(const ccs *A);
ccs * tril(const ccs *A);
int istril(const ccs *A);

ccs * chol_symbolic(const ccs *A);

/*! Sparse accumulator. Unpacked (dense) representation of a sparse vector */
typedef struct spa_struct {
  /** array of numerical values */
  double *val;
  /** boolean array to flag elements as non-zero */
  char *nz;
  /** indices of non-zero elements */
  int *idx;
  /** number of non-zeros */
  int nnz;
  /** length of vector */
  int n;
} spa;

/*! Allocates a spa structure of length <VAR>n</VAR> */
spa * alloc_spa(int_t n);

/*! frees a spa structure */
void free_spa(spa *s);

/*! Initializes a spa structure */
void init_spa(spa *s, const ccs *X, int col);
void spa_symb_axpy_uplo (const ccs *X, int col, spa *y, int j, char uplo);
void spa2compressed(spa *s, ccs *A, int col);


#endif
