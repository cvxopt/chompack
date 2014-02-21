/*!
  \file cliquetree.h
  \brief Data-structures and routines for creating clique-trees and
  chordal matrices.
  \author Joachim Dahl and Lieven Vandenberghe
  \version 1.1
  \date 19-11-2009

  This file contains routines and data-structures for creating for chordal
  matrices. An undirected graph G is chordal if every cycle of length
  greater than three has a chord, i.e., an edge joining nonconsecutive
  nodes of the cycle. A sparsity pattern \f$V\f$ defines an undirected
  graph \f$G_V\f$ with vertices \f$1,\dots,n\f$, and edges between
  nodes \f$i\f$ and \f$j\f$ if \f$(i, j) \in V , i \neq j\f$.

  A clique is a maximal subset of the nodes that defines a complete subgraph,
  i.e., all pairs of nodes in the clique are connected by an edge. The cliques
  can be represented by an undirected graph that has the cliques as its nodes,
  and edges between any two cliques with a nonempty intersection. We call this
  graph the clique graph associated with \f$G_V\f$ . We can also assign to
  every edge \f$(V_i, V_j)\f$ in the clique graph a weight equal to the number
  of nodes in the intersection \f$V_i \cap Vj\f$. A clique tree of a graph is
  a maximum weight spanning tree of its clique graph (as such a clique tree is
  nonunique).

  Assume that there are \f$l\f$ cliques \f$V_1, V_2, \dots, V_l\f$ in
  \f$G_V\f$. Then the set of nonzero entries is given by
  \f[\{(i, j) | (i, j) \in V\text{ or }(j, i) \in V \} = (V_1 \times V_1) \
  (V_2 \times V_2) \cup \cdots \cup (V_l \times V_l).\f]
  For a given clique tree we number the cliques (in reverse topological order)
  so that \f$V_1\f$ is the root of the tree and every parent in the tree has
  a lower index than its children. We define \f$S_1 = V_1, U_1 = \emptyset\f$
  and, for \f$i = 2, \dots, l\f$,
  \f[S_i = V_i \setminus (V_1 \cup V_2 \cup \cdots \cup V_{i-1}),
  \quad U_i = V_i \cap (V_1 \cup V_2 \cup \cdots V_{i−1}). \f]
  It can be shown that for a chordal graph
  \f[ S_i = V_i \setminus V_k, \quad U_i = V_i \cap V_k \f]
  where \f$V_k\f$ is the parent of \f$V_i\f$ in the clique tree.
  We assume that the nodes defining the chordal sparsity pattern are
  reordered using a perfect elimination order.

  As an example, consider the chordal matrix with sparsity pattern
  \f[
    \left[\begin{array}{ccccc}\circ & & & \circ & \circ\\
    & \circ & & \circ & \circ\\
    & & \circ & \circ & \circ\\
    \circ & \circ & \circ & \circ & \circ\\
    \circ & \circ & \circ & \circ & \circ
    \end{array}\right]
  \f]
  with nodes numbered from 0 to 4. We then have a clique-tree with three
  cliques (numbered in reverse topological order)
  \verbatim
       {U0, S0}
        /    \
  {U1, S1}  {U2, S2}
  \endverbatim
  with
  \f[ U_0 = \emptyset, \quad S_0 = \{0,3,4\}, \quad
  U_1 = \{3,4\}, \quad S_1=\{ 1\}, \quad U_2=\{3,4\}, \quad S_2=\{2\}.
  \f]
*/
#ifndef __CLIQUETREE__
#define __CLIQUETREE__

#include "chtypes.h"
#include "sparse.h"
#include <stdio.h>

/*! Clique object

 Stores indices \f$U_k\f$ amd \f$S_k\f$ for a clique, as
 well as pointers to parent and children cliques in the
 clique-tree.
*/
typedef struct clique {

  int *U;       /**< \f$U_k\f$ indices */
  int nu;       /**< size of \f$U_k\f$ */
  int *S;       /**< \f$S_k\f$ indices */
  int ns;       /**< size of \f$S_k\f$ */
  int *Uidx;    /**< indices mapping \f$U_k\f$ into its parent clique */
  int Uidx_ns;  /**< the first Uidx_ns elements of Uidx map to \f$S_{\text{par}(k)}\f$ */
  int ldu;      /**< leading dimension of \f$X_{U_kU_k}\f$ */

  struct clique *anc; /**< ancestor of clique in clique-forest */
  struct clique **Uanc;

  int nchildren; /**< number of children in clique-forest */
  int listidx;   /**< list index in reverse topological ordering */
  struct clique **children; /**< children in clique-forest */

} clique;

/*! Clique-forest object

 Represents a clique-forest as a list of the roots of the individual
 clique-trees. The structure also contains a list of the concatenation
 of all the clique-trees ordered in reverse topological order.
 */
typedef struct cliqueforest {

  int n;             /*!< dimension of chordal graph */
  int nRoots;        /*!< number of roots (or trees) in the forest */
  int nCliques;      /*!< number of cliques in the forest */
  int allocsize;     /*!< total memory size needed for the numerical values */
  struct clique **roots;  /*!< pointers to the roots of all trees */
  struct clique **list;   /*!< reverse topological ordering of all cliques */
  int_t *p;          /*!< permutation vector */
  int_t *ip;         /*!< inverse permutation */
  struct clique *K;  /*!< pointer to memory allocation */

} cliqueforest;

/*! Chordal matrix object

 Stores the numerical values of a chordal matrix object associated with a
 cliqueforest object.

 If <CODE>X</CODE> is a cliqueforest object, then the numerical values of
 the k'th clique (according the reverse topological ordering) can be accessed
 as

 <CODE>X->SS[k]</CODE>, <CODE>X->US[k]</CODE> and <CODE>X->UU[k]</CODE>

 with sizes

 <CODE>nS(F,k)*nS(F,k)</CODE>, <CODE>nU(F,k)*nS(F,k)</CODE> and
 <CODE>nU(F,k)*nU(F,k)</CODE>

 respectively.
 */
typedef struct {

  cliqueforest *F;  /*!< clique-forest specifying the sparsity pattern */
  double *data;     /*!< contiguous array storing the numerical values */
  double **US;      /*!< list of \f$X_{U_kS_k}\f$ blocks */
  double **SS;      /*!< list of \f$X_{S_kS_k}\f$ blocks */
  double **UU;      /*!< list of \f$X_{U_kU_k}\f$ blocks */
  void *F_py;       /*!< internal workspace used by Python interface */

} chordalmatrix;

/*! Vector object partioned conformally with a chordalmatrix object */
typedef struct {

  cliqueforest *F;  /*!< clique-forest giving the \f$\{U_k,S_k\}\f$ partitioning */
  double *data;     /*!< contiguous array storing the numerical values */
  double **U;       /*!< list of \f$x_{U_k}\f$ blocks */
  double **S;       /*!< list of \f$x_{S_k}\f$ blocks */
  int ncols;        /*!< number columns in the (block) vector */

} chordalvec;

/*! Creates a clique-forest from a chordal graph specified by a sparse
  matrix.

  \param[out]   F cliqueforest object
  \param[in]    X sparse lower-triangular matrix. The natural ordering must
  be a perfect elimination order for the graph for X.
  \param[in]    p permutation used for reordering X. If p is NULL, then a
  natural ordering is assumed.
 */
int cliqueforest_create(cliqueforest **F, const ccs *X, const int_t *p);

/*! Destroys a clique-forest object */
void cliqueforest_destroy(cliqueforest *F);

/*! Creates a chordal matrix by projecting a sparse matrix onto the
 sparsity pattern induced by a cliqueforest. Only the elements
 in the lower-triangular part of X are used.
 */
chordalmatrix * ccs_to_chordalmatrix (const cliqueforest *F, const ccs *X) ;

/*! Creates a chordal matrix by projecting a dense matrix (stored
 in column-order mode) onto the sparsity pattern induced by a cliqueforest.
 Only the elements in the lower-triangular part of X are used.
 */
chordalmatrix * dense_to_chordalmatrix(const cliqueforest *F, const double *x);

/*! copies a chordal matrix object to a sparse matrix */
ccs * chordalmatrix_to_ccs (const chordalmatrix *A);

/*! copies a chordal factor to a sparse matrix */
ccs * chordalfactor_to_ccs (const chordalmatrix *A);

/*! destroys a chordal matrix object */
void chordalmatrix_destroy(chordalmatrix *X);

/*! creates a copy of a chordal matrix object. */
chordalmatrix * chordalmatrix_copy(const chordalmatrix *A);

/*! Internal routine for copying data between parent and child cliques. */
void cliquecopy(chordalmatrix *X, const clique *child, int toparent);

/*! Internal routine for copying data between parent and child cliques. */
void cliquecopy_nmf(chordalmatrix *X, const clique *child, int toparent, double *src);

/*! Allocates a block vector partitioned conformally with a chordal matrix object. */
chordalvec * alloc_chordalvec(const cliqueforest *F, int ncols);

/*! Internal routine for copying a dense (block) vector into a previously
  allocated chordal vector object.

  \param[in,out] X block vector partitioned conformally with a chordal matrix.
  \param[in]  x dense matrix stored in column-order mode.
  \paran[in]  p row-permutation for x. If it is NULL, then no permutation is applied.
  \param[in] trans 'N' or 'T'. If it is 'T' then transpose of x is used.
 */
void copy_dense_to_chordalvec(chordalvec *X, const double *x, const int_t *p, char trans);

/*! Internal routine for copying parts of a sparse matrix into a chordal vector object.

  \param[in,out] X block vector partitioned conformally with a chordal matrix.
  \param[in]  n number of columns in block.
  \param[in]  x sparse matrix.
  \param[in]  coloffs column offset in x.
  \param[in]  s spa structure for with the same length as X
  \paran[in]  p row-permutation for x. If it is NULL, then no permutation is applied.
 */
void sparse_to_chordalvec(chordalvec *X, int n, ccs *x, int coloffs, spa *s, int_t *p);

/*! Internal routine for copying a dense matrix into a chordal vector object.

  \param[in] F clique-forest giving the partitioning of the chordal vector
  \param[in] x dense matrix stored in column-order mode.
  \param[in] ncols number of columns in x
  \paran[in] p row-permutation for x. If it is NULL, then no permutation is applied.
  \returns   a new chordalvec object
*/
chordalvec * dense_to_chordalvec(const cliqueforest *F, const double *x, int ncols, const int_t *p);

/*! destroys a chordal vector */
void chordalvec_destroy(chordalvec *X);

/*! copies a chordal vector into a dense matrix

  \param[in] X chordal vector
  \param[out] Y dense matrix
  \paran[in] p row-permutation for Y. If it is NULL, then no permutation is applied.
 */
void chordalvec_to_dense (const chordalvec *X, double *Y, const int_t *p);

/*! Internal routine for copying data between parent and child cliques. */
void cliquecopy_vec(chordalvec *X, const clique *child, int toparent);

/*! Internal routine for copying data between parent and child cliques. */
void cliquecopy_vec_nmf(chordalvec *X, const clique *child, int toparent, double *src);

#define nU(F, k)     (F->list[k]->nu)     /*!< Number of elements in \f$U_k\f$ */
#define  U(F, k)     (F->list[k]->U)      /*!< List of indices \f$U_k\f$ */
#define nS(F, k)     (F->list[k]->ns)     /*!< Number of elements in \f$S_k\f$ */
#define  S(F, k)     (F->list[k]->S)      /*!< List of indices \f$S_k\f$ */

/** \cond */
#define  Uidx(F, k)  (F->list[k]->Uidx)
#define FREE(O) if (O) { free(O); O = NULL; }
/** \endcond */

#endif
