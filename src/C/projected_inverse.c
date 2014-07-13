#include "chompack.h"

int projected_inverse(const int_t n,         // order of matrix
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

  int nn,na,nj,offset,info,i,j,k,l,N,ki,nup=0;
  double * restrict U;
  double dOne=1.0,dNegOne=-1.0,dZero=0.0;
  char cL='L',cT='T',cN='N';

  U = upd;   // pointer to top of update storage

  for (ki=nsn-1;ki>=0;ki--) {
    k = snpost[ki];
    nn = snptr[k+1]-snptr[k];
    na = relptr[k+1]-relptr[k];
    nj = na + nn;

    // invert factor of D_{Nk,Nk}
    dtrtri_(&cL, &cN, &nn, blkval+blkptr[k], &nj, &info);
    if (info) return info;

    // zero-out strict upper triangular part of {Nj,Nj} block (just in case!)
    for (j=1;j<nn;j++) {
      for (i=0;i<j;i++) blkval[blkptr[k]+j*nj+i] = 0.0;
    }

    // compute inv(D_{Nk,Nk}) (store in 1,1 block of frontal matrix)
    dsyrk_(&cL, &cT, &nn, &nn, &dOne, blkval+blkptr[k], &nj, &dZero, fws, &nj);

    // if supernode k is not a root node:
    if (na>0) {
      // copy update matrix to 2,2 block of frontal matrix
      nup--;
      U -= upd_size[nup]*upd_size[nup];
      dlacpy_(&cL, &na, &na, U, &na, fws+nn*nj+nn, &nj);

      // compute S_{Ak,Nk} = -Vk*L_{Ak,Nk}; store in 2,1 block of F
      dsymm_(&cL, &cL, &na, &nn, &dNegOne, fws+nn*nj+nn, &nj,
	     blkval+blkptr[k]+nn, &nj, &dZero, fws+nn, &nj);

      // compute S_nn = inv(D_{Nk,Nk}) - S_{Ak,Nk}'*L_{Ak,Nk}; store in 1,1 block of F
      dgemm_(&cT, &cN, &nn, &nn, &na, &dNegOne, fws+nn, &nj,
	     blkval+blkptr[k]+nn, &nj, &dOne, fws, &nj);
    }

    // extract update matrices if supernode k has any children
    for (l=chptr[k];l<chptr[k+1];l++) {
      offset = relptr[chidx[l]];
      N = relptr[chidx[l]+1]-offset;
      upd_size[nup++] = N;
      for (j=0; j<N; j++) {
				for (i=j; i<N; i++) {
	  			U[N*j+i] = fws[nj*relidx[offset+j]+relidx[offset+i]];
				}
      }
      U += N*N;
    }
    // copy S_{Jk,Nk} (i.e., 1,1 and 2,1 blocks of frontal matrix) to blkval
    dlacpy_(&cL, &nj, &nn, fws, &nj, blkval+blkptr[k], &nj);
  }
  return 0;
}
