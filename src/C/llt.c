#include "chompack.h"

void llt(const int_t n,         // order of matrix
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

  int nn,na,nj,offset,i,j,k,ki,l,N,nup=0;
  double * restrict U;
  double dOne=1.0,dZero=0.0;
  char cL='L',cR='R',cN='N';

  U = upd;   // pointer to top of update storage

  for (ki=0;ki<nsn;ki++) {
    k = snpost[ki];
    nn = snptr[k+1]-snptr[k];
    na = relptr[k+1]-relptr[k];
    nj = na + nn;        

    // compute [I; L_{Ak,Nk}]*D_k*[I;L_{Ak,Nk}]'; store in frontal workspace
    dtrmm_(&cR, &cL, &cN, &cN, &na, &nn, &dOne, blkval+blkptr[k], &nj, blkval+blkptr[k]+nn, &nj);
    dsyrk_(&cL, &cN, &nj, &nn, &dOne, blkval+blkptr[k], &nj, &dZero, fws, &nj);
    
    // add update matrices to frontal matrix
    for (l=chptr[k+1]-1;l>=chptr[k];l--) {
      nup--;
      U -= upd_size[nup]*upd_size[nup];
      // extend-add
      offset = relptr[chidx[l]];
      N = relptr[chidx[l]+1] - offset;
      for (j=0; j<N; j++) {
	for (i=j; i<N; i++)
	  fws[nj*relidx[offset+j]+relidx[offset+i]] += U[N*j+i];
      }
    }

    // if supernode k is not a root node, push update matrix onto stack
    if (na > 0) {   
      upd_size[nup++] = na;
      dlacpy_(&cL, &na, &na, fws+nn*nj+nn, &nj, U, &na);
      U += na*na;
    }

    // copy the leading nn columns of frontal matrix to blkval
    dlacpy_(&cL, &nj, &nn, fws, &nj, blkval+blkptr[k], &nj);
  }

  return;
}
