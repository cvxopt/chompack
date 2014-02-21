#ifndef __CHORDAL__
#define __CHORDAL__

#ifdef __cplusplus
extern "C" {
#endif

#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

#include "chtypes.h"
#include "cliquetree.h"
#include "adjgraph.h"
#include "sparse.h"
#include "pf_cholesky.h"

#define CHOMPACK_OK                  0
#define CHOMPACK_NOMEMORY            1
#define CHOMPACK_FACTORIZATION_ERR   2
#define CHOMPACK_SOLVE_ERR           3
#define CHOMPACK_NONCHORDAL          4

int cholesky(chordalmatrix *X);
int solve(const chordalmatrix *X, double *B, int n, int sys);

int llt(chordalmatrix *X);
int partial_inv(chordalmatrix *X);

chordalmatrix * completion(const chordalmatrix *X, int *info);
double dot(const chordalmatrix *X, const chordalmatrix *Y);
void axpy(const chordalmatrix *X, chordalmatrix *Y, double a);
void scal(double a, chordalmatrix *X);
int hessian_factor
(const chordalmatrix *X, const chordalmatrix *Y, chordalmatrix **dX, int adj, int inv, int m);

double logdet(const chordalmatrix *X);

void syr1(chordalmatrix *X, double *y, double a, double b);
void syr2(chordalmatrix *X, double *y, double *z, double a, double b);

adjgraph *maxchord(const adjgraph *A, const int_t startvertex, int_t **order);

#ifdef __cplusplus
}
#endif

#endif
