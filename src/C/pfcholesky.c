#include <math.h>
#include "chompack.h"

void ddrsv(const int *n, const char *trans, double * restrict v, double * restrict l, double * restrict b, double * restrict x) {
  /*
    Solves a system of the form L*y = x or L.T*y = x where L is the
    Cholesky factor of a matrix of the form I + a*v*v.T. On exit, x
    contains the solution y.
   */
  int k;
  double s=0;
  if (*trans=='N') {
    for (k=0;k<*n;k++) {
      x[k] = (x[k]-v[k]*s)/l[k];
      s += x[k]*b[k];
    }
  }
  else if (*trans=='T') {
    for (k=*n-1;k>=0;k--) {
      x[k] = (x[k]-b[k]*s)/l[k];
      s += x[k]*v[k];
    }
  }
  return;
}

void ddrmv(const int *n, const char *trans, double * restrict v, double * restrict l, double * restrict b, double * restrict x) {
  /*
    Computes the matrix-vector product x := L*x or x := L.T*x where L
    is the Cholesky factor of a matrix of the form I + a*v*v.T.
   */
  int k;
  double tmp,s=0;
  if (*trans=='N') {
    for (k=0;k<*n;k++) {
      tmp = x[k]*b[k];
      x[k] = l[k]*x[k]+v[k]*s;
      s += tmp;
    }
  }
  else if (*trans=='T') {
    for (k=*n-1;k>=0;k--) {
      tmp = x[k]*v[k];
      x[k] = l[k]*x[k]+b[k]*s;
      s += tmp;
    }
  }
  return;
}


int dpftrf(const int *n, const int *k, double * restrict a, double * restrict V, const int *ldv, double * restrict L, const int *ldl, double * restrict B, const int *ldb) {
  /*
    Computes product-form Cholesky factorization of a matrix I +
    V*diag(a)*V.T of order n and with V n-by-m, i.e,

      I + V*diag(a)*V.T = L_1*L_2*...*L_k*L_k'*...*L_2'*L_1'.

    The arrays L and B must each be of length (at least) n*k.
   */
  int i,j;
  double tmp, alpha;
  double *v,*l,*b;
  char cN='N';
  
  for (i=0;i<*k;i++) {
    
    v = V+i*(*ldv); 
    b = B+i*(*ldb); 
    l = L+i*(*ldl);
    alpha = -a[i];
    
    for (j=0;j<*n;j++) {
      tmp = 1.0-alpha*v[j]*v[j];
      if (tmp<=0) return -1;
      l[j] = sqrt(tmp);
      b[j] = -alpha*v[j]/l[j];
      alpha += b[j]*b[j];
    }
    
    for (j=i+1;j<*k;j++) {
      ddrsv(n,&cN,v,l,b,V+j*(*ldv));
    }

  }
  return 0;
}

void dpfsv(const int *n, const int *k, const char *trans, double * restrict V, const int *ldv, double * restrict L, const int *ldl, double * restrict B, const int *ldb, double * restrict x) {  
  /*
    Solves a system 

        L_1*L_2*...*L_k*y = x      if trans == 'N'

    or 

        L_k'*...*L_2'*L_1'*y = x   if trans == 'T'
 
    where L_1*L_2*...*L_k*L_k'*...*L_2'*L_1' is the product-form
    Cholesky-factorization of a matrix I + V*diag(a)*V.T of order
    n. On exit, x contains the solution y.
   */

  int i;  
  if (*trans == 'N') {
    for (i=0;i<*k;i++) ddrsv(n,trans,V+i*(*ldv),L+i*(*ldl),B+i*(*ldb),x);
  }
  else if (*trans == 'T') {
    for (i=*k-1;i>=0;i--) ddrsv(n,trans,V+i*(*ldv),L+i*(*ldl),B+i*(*ldb),x);
  }
  return;
}
		
void dpfmv(const int *n, const int *k, const char *trans, double * restrict V, const int *ldv, double * restrict L, const int *ldl, double * restrict B, const int *ldb, double * restrict x) {  
  /*
    Computes the matrix-vector product

        x := L_1*L_2*...*L_k*x      if trans == 'N'

    or 

        x := L_k'*...*L_2'*L_1'*x   if trans == 'T'
 
    where L_1*L_2*...*L_k*L_k'*...*L_2'*L_1' is the product-form
    Cholesky-factorization of a matrix I + V*diag(a)*V.T of order
    n.
   */
  int i;  
  if (*trans == 'N') {
    for (i=*k-1;i>=0;i--) ddrmv(n,trans,V+i*(*ldv),L+i*(*ldl),B+i*(*ldb),x);
  }
  else if (*trans == 'T') {
    for (i=0;i<*k;i++) ddrmv(n,trans,V+i*(*ldv),L+i*(*ldl),B+i*(*ldb),x);
  }
  return;
}
