/*
 * Copyright (c) 2009 J. Dahl and L. Vandenberghe.
 *
 * Example program for CHOMPACK.
 * Computes the partial inverse of the chordal matrix
 *
 *        [10,  0,  0,  0,  1]
 *        [ 0, 10,  0,  0,  1]
 *   A =  [ 0,  0, 10,  0,  1]
 *        [ 0,  0,  0, 10,  1]
 *        [ 1,  1,  1,  1, 10]
 */

#include <stdio.h>
#include "chompack.h"

int main() {

  int j, k;
  double Aval[] = {10.0, 1.0, 10.0, 1.0, 10.0, 1.0, 10.0, 1.0, 10.0};
  int_t  Acol[] = {0, 2, 4, 6, 8, 9};
  int_t  Arow[] = {0, 4, 1, 4, 2, 4, 3, 4, 5};

  ccs *A, *Y;
  cliqueforest *F;
  chordalmatrix *X;

  A = alloc_ccs(5, 5, 9);
  for (j=0; j<9; j++) {
    A->rowind[j] = Arow[j];
    ((double *)A->values)[j] = Aval[j];
  }

  for (j=0; j<6; j++)
    A->colptr[j] = Acol[j];

  printf("creating cliquetree\n");
  cliqueforest_create(&F, A, NULL);

  printf("creating chordal matrix\n");
  X = ccs_to_chordalmatrix (F, A);

  printf("performing factorization\n");
  cholesky(X);

  printf("computing partial inverse\n");
  partial_inv(X);

  Y = chordalmatrix_to_ccs(X);

  printf("partial inverse:\n");
  for (j=0; j<Y->ncols; j++)
    for (k=Y->colptr[j]; k<Y->colptr[j+1]; k++)
      printf("(%i,%i): % 3.2e\n", (int)Y->rowind[k], j, ((double *)Y->values)[k]);

  chordalmatrix_destroy(X);
  cliqueforest_destroy(F);
  free_ccs(Y);

  return 0;

}
