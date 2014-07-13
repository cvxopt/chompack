#include "chompack.h"

int cholesky(const int_t n,         // order of matrix
	     const int_t nsn,       // number of supernodes/cliques
	     const int_t *snpost,   // post-ordering of supernodes
	     const int_t *snptr,    // supernode pointer
	     const int_t *relptr,
	     const int_t *relidx,
	     const int_t *chptr,
	     const int_t *chidx,
	     const int_t *blkptr,
	     double * restrict blkval,
	     double * restrict fws,  // frontal matrix workspace
	     double * restrict upd,  // update matrix workspace
	     int_t * restrict upd_size
	     ) {

  int nn,na,nj,offset,info,i,j,k,ki,l,N,nup=0;
  double * restrict U;
  int iOne=1;
  double dOne=1.0,dNegOne=-1.0;
  char cL='L',cT='T',cR='R',cN='N';

  U = upd;   // pointer to top of update storage

  for (ki=0;ki<nsn;ki++) {
    k = snpost[ki];
    nn = snptr[k+1]-snptr[k];
    na = relptr[k+1]-relptr[k];
    nj = na + nn;

    // build frontal matrix
    dlacpy_(&cL, &nj, &nn, blkval+blkptr[k], &nj, fws, &nj);
    for (j=nn;j<nj;j++) {
      for (i=j;i<nj;i++) {
	fws[nj*j+i] = 0.0; // zero out (2,2) block of frontal matrix
      }
    }

    // add update matrices to frontal matrix
    for (l=chptr[k+1]-1;l>=chptr[k];l--) {
      nup--;
      U -= upd_size[nup]*upd_size[nup];
      // extend-add
      offset = relptr[chidx[l]];
      N = relptr[chidx[l]+1] - offset;
      for (j=0; j<N; j++) {
	for (i=j; i<N; i++) {
	  fws[nj*relidx[offset+j]+relidx[offset+i]] += U[N*j+i];
	}
      }
    }

    // factor L_{Nk,Nk}
    dpotrf_(&cL, &nn, fws, &nj, &info);
    if (info) return info;

    // if supernode k is not a root node, compute and push update matrix onto stack
    if (na > 0) {
      // compute L_{Ak,Nk} := A_{Ak,Nk}*inv(L_{Nk,Nk}')
      dtrsm_(&cR, &cL, &cT, &cN, &na, &nn, &dOne, fws, &nj, fws+nn, &nj);

      // compute Uk = Uk - L_{Ak,Nk}*inv(D_{Nk,Nk})*L_{Ak,Nk}'
      if (nn == 1) {
	dsyr_(&cL, &na, &dNegOne, fws+nn, &iOne, fws+nn*nj+nn, &nj);
      }
      else {
	dsyrk_(&cL, &cN, &na, &nn, &dNegOne, fws+nn, &nj, &dOne, fws+nn*nj+nn, &nj);
      }

      // compute L_{Ak,Nk} := L_{Ak,Nk}*inv(L_{Nk,Nk})
      dtrsm_(&cR, &cL, &cN, &cN, &na, &nn, &dOne, fws, &nj, fws+nn, &nj);

      // copy update matrix to stack
      upd_size[nup++] = na;
      dlacpy_(&cL, &na, &na, fws+nn*nj+nn, &nj, U, &na);
      U += na*na;
    }

    // copy the leading nn columns of frontal matrix to blkval
    dlacpy_(&cL, &nj, &nn, fws, &nj, blkval+blkptr[k], &nj);
  }
  return 0;
}
