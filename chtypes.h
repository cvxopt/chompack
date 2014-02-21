#ifndef int_t         /* defined by CVXOPT */

#ifdef DLONG
#define int_t long int
#else
#define int_t int
#endif

#define DOUBLE 1

typedef struct {
  void  *values;      /* value list */
  int_t *colptr;      /* column pointer list */
  int_t *rowind;      /* row index list */
  int_t nrows, ncols; /* number of rows and columns */
  int   id;           /* not currently used */
} ccs;

#endif
