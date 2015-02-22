#include "chompack.h"

double dot(int_t *Nsn, int_t *snptr, int_t *sncolptr, int_t *blkptr, double * restrict blkval_x, double * restrict blkval_y) {

  double val = 0.0;
  int inc=1,n;
  int_t j,k,offset,nk,ck;
  
  for (k=0;k<*Nsn;k++) {
    offset = blkptr[k];
    nk = snptr[k+1]-snptr[k];
    ck = sncolptr[k+1]-sncolptr[k];
    n = ck;
    for (j=0;j<nk;j++) {
      val -= blkval_x[offset]*blkval_y[offset];
      val += 2.0*ddot_(&n, blkval_x+offset, &inc, blkval_y+offset, &inc);
      n -= 1;
      offset += ck + 1;
    }
  }
  return val;
}

