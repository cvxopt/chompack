#include "chompack.h"

int completion(const int_t n,         // order of matrix
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
	       int_t * restrict upd_size,
	       int factored_updates) {

  int nn,na,nj,offset,info,N,i,j,k,ki,l,nup=0,iOne=1;
  double * restrict U, * restrict ws=NULL;
  double dOne=1.0,dNegOne=-1.0;
  char cL='L',cU='U',cT='T',cN='N';
  char *trL1, *trL2;

  U = upd;   // pointer to top of update storage

  for (ki=nsn-1;ki>=0;ki--) {
    k = snpost[ki];
    nn = snptr[k+1]-snptr[k];
    na = relptr[k+1]-relptr[k];
    nj = na + nn;

    // copy L_{Jk,Nk} to leading columns of F
    dlacpy_(&cL, &nj, &nn, blkval+blkptr[k], &nj, fws, &nj);

    // if supernode k is not a root node:
    if (na > 0) {
      // copy update matrix to 2,2 block of frontal matrix
      nup--;
      U -= upd_size[nup]*upd_size[nup];
      dlacpy_(&cL, &na, &na, U, &na, fws+nn*nj+nn, &nj);
    }

    if ((chptr[k+1]-chptr[k]>0) && (factored_updates)) {
      ws = malloc(na*(na+1)*sizeof(double)); // allocate workspace
    }

    // extract update matrices if supernode k has any children
    for (l=chptr[k];l<chptr[k+1];l++) {

      offset = relptr[chidx[l]];
      N = relptr[chidx[l]+1]-offset;
      upd_size[nup++] = N;

      if (factored_updates) {
	info = update_factor(relidx+offset, &nn, &na, U, &N, fws, &nj, ws);
	if (info) {
	  free(ws);
	  return info;
	}
      }
      else {
	/* extract unfactored update */
	for (j=0; j<N; j++) {
	  for (i=j; i<N; i++) {
	    U[N*j+i] = fws[nj*relidx[offset+j]+relidx[offset+i]];
	  }
	}
      }
      U += N*N;
    }

    // free workspace if node has any children and factored_updates
    if ((chptr[k+1]-chptr[k]>0) && (factored_updates)) {
      free(ws);
    }

    // if supernode k is not a root node:
    if (na>0) {
      if (factored_updates) {
        // In this case we have Vk = Lk'*Lk
	trL1 = &cT;
	trL2 = &cN;
      }
      else {
	// factorize Vk
	dpotrf_(&cL, &na, fws+nj*nn+nn, &nj, &info);
	if (info) return info;
	// In this case we have Vk = Lk*Lk'
	trL1 = &cN;
	trL2 = &cT;
      }
      // compute L_{Ak,Nk} and inv(D_{Nk,Nk}) = S_{Nk,Nk} - S_{Ak,Nk}'*L_{Ak,Nk}
      dtrtrs_(&cL, trL1, &cN, &na, &nn, fws+nj*nn+nn, &nj, blkval+blkptr[k]+nn, &nj, &info);
      if (info) return info;
      dsyrk_(&cL, &cT, &nn, &na, &dNegOne, blkval+blkptr[k]+nn, &nj, &dOne, blkval+blkptr[k], &nj);
      dtrtrs_(&cL, trL2, &cN, &na, &nn, fws+nj*nn+nn, &nj, blkval+blkptr[k]+nn, &nj, &info);
      if (info) return info;
      for (j=0;j<nn;j++) dscal_(&na, &dNegOne, blkval+blkptr[k] + j*nj + nn, &iOne);
    }

    // factorize inv(D_{Nk,Nk}) as R*R' so that D_{Nk,Nk} = L*L' with L = inv(R)'
    for (j=0;j<nn;j++) { // reverse-copy
      for (i=j;i<nn;i++) {
	fws[(nn-j)*nn-1-i] = blkval[blkptr[k]+j*nj+i];
      }
    }
    dpotrf_(&cU, &nn, fws, &nn, &info);
    if (info) return info;
    for (j=0;j<nn;j++) { // reverse-copy
      for (i=j;i<nn;i++) {
	blkval[blkptr[k]+j*nj+i] = fws[(nn-j)*nn-1-i];
      }
    }

    // compute L = inv(R')
    dtrtri_(&cL, &cN, &nn, blkval+blkptr[k], &nj, &info);
    if (info) return info;
  }
  return 0;
}
