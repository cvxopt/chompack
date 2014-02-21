/*!
  \file pdf_cholesky.h
  \brief Product-form Cholesky factorization
  \author Joachim Dahl and Lieven Vandenberghe
  \version 1.1
  \date 19-11-2009

  Routines for solving
  \f[ (A+BB^T)x = b \f]
  using a product-form Cholesky factorization where \f$A\f$ is symmetric
  positive semidefinte and \f$B\f$ is a dense (low-rank) matrix.
  The product-form factorization is
  \f[ A+BB^T = PLDL^TP^T \f]
  where \f$L\f$ is a unit-diagonal lower-triangular matrix, \f$D\f$
  is a positive diagonal matrix and \f$P\f$ is permutation matrix
  for reducing fill-in in the factor of \f$A\f$. The factor \f$L\f$
  is never stored explicit; instead it is implicitly given by the
  factorization of \f$A\f$ combined with a series of low-rank updates
  associated with the dense low-rank term \f$BB^T\f$.
*/
#ifndef __PRODUCTFORM_MF_CHOL__
#define __PRODUCTFORM_MF_CHOL__

#include "chompack.h"

/*! Product-form factorization object.

 Factor object for storing the product-form factorization of
 \f[ A+BB^T \f]
 where \f$A\f$ is symmetric positive semidefinite and \f$B\f$ is a
 dense low-rank matrix.
*/
typedef struct pf_factor_struct
{
  int n;              /**< dimension of A */
  int ndense;         /**< ndense number of columns in B */
  int_t *P;           /**< fill-reducing permutation sequence for A */
  int_t *Pi;          /**< fill-reducing inverse permutation sequence for A */
  cliqueforest *Asymbolic; /**< object representing the chordal sparsity pattern of the Cholesky factor of A */
  chordalmatrix *A;   /**< object with the numerical values of the factor of A */
  chordalvec *p;      /**< parameters of low-rank factors of B */
  chordalvec *beta;   /**< parameters of low-rank factors of B */
  chordalvec *work;   /**< work-storage */
  double *work2;      /**< work-storage */
  double droptol;     /**< zero-pivot drop-tolerance used in factorization */
  int is_numeric_factor; /**< indicates whether object represents a numerical or symbolic factorization */
} pf_factor;

/*! Semidefinite \f$LDL^T\f$ factorization of a dense matrix.

  Computes a factorization \f$X=LDL^T\f$ where \f$L\f$ is unit-diagonal
  lower triangular and \f$D\f$ is diagonal. On exit, the lower-triangular
  part of X is overwritten with strictly lower-triangular part of L and the diagonal of D.

  \param[in,out] X array stored in column-order-mode
  \param[in] ld leading dimension
  \param[in] n  dimension of X
  \param[in] blksize blocksize used in factorization.
  \param[in] droptol drop-tolerance for zero pivot-elements
  \return CHOMPACK_OK or CHOMPACK_FACTORIZATION_ERR.
*/
int ldl(double *X, int ld, int n, int blksize, double droptol);

/*! Symbolic factorization.

  Performs symbolic factorization phase (computes sparsity-pattern
  and allocates data-structures).

  \param[in] A sparse semidefinite matrix. Must be symmetric
  \param[in] p permutation vector. If p is NULL, then the AMD reordering is used.
  \param[in] ndense number of (dense) columns in product-form factorization (i.e., in B)
  \param[in] droptol drop-tolerance for zero pivot-elements
  \param[in] output if non-zero the routine prints log output
  \return a product-form factorization object, or NULL if insufficient memory
 */
pf_factor * pf_symbolic(const ccs *A, int_t *p, int ndense, double droptol, int output);

/*! Numeric factorization.

  Performs a numerical factorization of
  \f[ A+BB^T \quad \text{if} \quad \text{trans=='N'} \f]
  or
  \f[ A+B^TB \quad \text{if} \quad \text{trans=='T'} \f]
  using a symbolic factorization created by a call to pf_symbolic().
  On entry <VAR>f</VAR> contains either a symbolic of numeric
  factorization object, and on exit <VAR>f</VAR> is updated with
  the newly computed numerical factorization. The sparsity pattern of
  <VAR>A</VAR> is assumed to be a subset of the sparsity pattern used
  in the previous call to pf_symbolic(); elements outside the symbolic
  sparsity pattern are ignored.

  \param[in,out] f factorization object obtained by a call to pf_symbolic()
  \param[in] A positive semidefinite sparse matrix. Only the lower triangular part is used,
  \param[in] B array storing the dense matrix B in column-order mode.
  \param[in] trans indicates whether the dense low-rank term is \f$BB^T\f$ (trans=='N') or \f$B^TB\f$ (trans=='T')
  \return CHOMPACK_OK or CHOMPACK_FACTORIZATION_ERR
 */
int pf_numeric(pf_factor *f, const ccs *A, const double *B, char trans);

/** Frees a factor object created by pf_symbolic() */
void free_pf_factor(pf_factor *f);

/*!  Solves a linear set of equations with dense right-hand-sides.

 Solution of a set of linear equations using a product-form factorization
 \f[ A+BB^T = PLDL^TP^T \f]
 with a dense right-hand-side.

 Solves
 \f{align*} PLDL^TP^Tx & = b \quad \text{if sys==0}\\
   PLD^{1/2}x & = b \quad \text{if sys==1}\\
   D^{1/2}L^TP^Tx  &= b \quad \text{if sys==2}
 \f}
 and overwrites \f$b\f$ with the solution \f$x\f$.

 \param[in] f numerical factorization object obtained by a call to pf_numeric()
 \param[in,out] b dense matrix stored in column-order mode
 \param[in] n number of right-hand-sides (columns in b)
 \param[in] sys integer
 \return CHOMPACK_OK or CHOMPACK_FACTORIZATION_ERR
*/
int pf_solve(pf_factor *f, double *b, int n, int sys);

/*!  Solves a linear set of equations with sparse right-hand-sides.

 Solution of a set of linear equations using a product-form factorization
 \f[ A+BB^T = PLDL^TP^T \f]
 with a dense right-hand-side.

 Solves
 \f{align*} PLDL^TP^Tx & = b \quad \text{if sys==0}\\
   PLD^{1/2}x & = b \quad \text{if sys==1}\\
   D^{1/2}L^TP^Tx  &= b \quad \text{if sys==2}.
 \f}
 The routine allocates a new object for the sparse solution \f$x\f$.

 \param[in] f numerical factorization object obtained by a call to pf_numeric()
 \param[in] b sparse matrix
 \param[out] x sparse matrix
 \param[in] n number of right-hand-sides (columns in b)
 \param[in] sys integer
 \return CHOMPACK_OK, CHOMPACK_FACTORIZATION_ERR or CHOMPACK_NOMEMORY
*/
int pf_spsolve(pf_factor *f, ccs *b, ccs **x, int sys, int blksize);

#endif
