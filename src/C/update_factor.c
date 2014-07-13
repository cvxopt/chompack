#include "chompack.h"

int update_factor(const int_t *ri, int *nn, int *na, double * restrict u, int *ldu, double * restrict f, int *ldf, double *ws) {

  int N1,N2,i,j,ii,jj,offset,info,m,iOne=1;
  double dOne=1.0,dNegOne=-1.0,dtmp=0.0,tau=0.0;
  char cL='L',cU='U',cT='T',cN='N';

  /*
    Compute:
    N1 = number of entries in ri such that ri[k] < nn
    N2 = number of entries in ri such that ri[k] >= nn
  */
  N1 = 0;
  for (i=0;i<(*ldu);i++) {
    if (ri[i] >= (*nn)) break;
    else N1 += 1;
  }
  N2 = (*ldu)-N1;

  // extract A and B from frontal matrix; copy to leading N1 columns of upd. mat.
  for (j=0;j<N1;j++) {
    for (i=j;i<(*ldu);i++) {
      u[(*ldu)*j+i] = f[(*ldf)*ri[j]+ri[i]];
    }
  }

  // check if updating is necessary
  if (N2 > 0) {
    // extract C = F[nn:,r[N1:]]
    for (j=0;j<N2;j++) {
      jj = ri[N1+j];
      for (ii=jj;ii<(*ldf);ii++) {
        ws[(*na)*j+ii-(*nn)] = f[(*ldf)*jj+ii];
      }
    }

    // reduce C to lower triangular from
    for (j=0;j<N2;j++) {
      jj = ri[(*ldu)-1-j]-(*nn);  // index of column in C (ws)
      ii = ((*na)-jj)-(j+1);            // number of nonzeros to zero out
      if (ii == 0) continue;

      // compute and apply Householder reflector
      m = ii + 1;
      dlarfg_(&m, ws+(*na)*(N2-j)-1-j, ws+(*na)*(N2-j)-1-j-ii, &iOne, &tau);
      dtmp = ws[(*na)*(N2-j)-1-j];
      ws[(*na)*(N2-j)-1-j] = 1.0;
      i = N2-1-j;
      dlarfx_(&cL, &m, &i, ws+(*na)*(N2-j)-1-j-ii, &tau, ws+(*na)-1-j-ii, na, ws+(*na)*(*na));
      ws[(*na)*(N2-j)-1-j] = dtmp;
    }

    // copy lower triangular matrix from ws to 2,2 block of U
    dlacpy_(&cL, &N2, &N2, ws+(*na)-N2, na, u+((*ldu)+1)*N1, ldu);

    // compute L_{21} by solving L_{22}'*L_{21} = B
    dtrtrs_(&cL, &cT, &cN, &N2, &N1, u+N1*(*ldu)+N1, ldu, u+N1, ldu, &info);
    if (info) return info;

    // compute A - Li_{21}'*Li_{21}
    dsyrk_(&cL, &cT, &N1, &N2, &dNegOne, u+N1, ldu, &dOne, u, ldu);
  }

  /* Compute Li_{11}  (reverse -- factorize -- reverse) */
  // reverse 1,1 block
  offset = ((*ldu)+1)*(N1-1);
  for (j=0;j<N1/2;j++) {
    for (i=0;i<N1;i++) {
      dtmp = u[offset-j*(*ldu)-i];
      u[offset-j*(*ldu)-i] = u[(*ldu)*j+i];
      u[(*ldu)*j+i] = dtmp;
    }
  }
  if (N1 % 2) { // nn is odd
    j = N1/2;
    for (i=0;i<N1/2;i++) {
      dtmp = u[offset-j*(*ldu)-i];
      u[offset-j*(*ldu)-i] = u[(*ldu)*j+i];
      u[(*ldu)*j+i] = dtmp;
    }
  }

  // factorize 1,1 block
  dpotrf_(&cU, &N1, u, ldu, &info);
  if (info) return info;

  // reverse 1,1 block
  offset = ((*ldu)+1)*(N1-1);
  for (j=0;j<N1/2;j++) {
    for (i=0;i<N1;i++) {
      dtmp = u[offset-j*(*ldu)-i];
      u[offset-j*(*ldu)-i] = u[(*ldu)*j+i];
      u[(*ldu)*j+i] = dtmp;
    }
  }
  if (N1 % 2) { // nn is odd
    j = N1/2;
    for (i=0;i<N1/2;i++) {
      dtmp = u[offset-j*(*ldu)-i];
      u[offset-j*(*ldu)-i] = u[(*ldu)*j+i];
      u[(*ldu)*j+i] = dtmp;
    }
  }

  return 0;

}
