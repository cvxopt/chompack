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

  int nn,na,nj,offset,info,N,N1,N2,i,ii,j,jj,k,ki,l,m,nup=0,iOne=1;
  double * restrict U, * restrict ws=NULL;
  double dOne=1.0,dNegOne=-1.0,dtmp=0.0,tau=0.0;
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
	/* Compute:
	     N1 = number of entries in relidx such that ri < nn
	     N2 = number of entries in relidx such that ri >= nn 
	*/
	N1 = 0;
	for (i=0;i<N;i++) {
	  if (relidx[offset+i] >= nn) break;
	  else N1 += 1;
	}
	N2 = N-N1;
	
	// extract A and B from frontal matrix; copy to leading N1 columns of U
	for (j=0;j<N1;j++) {
	  for (i=j;i<N;i++) {
	    U[N*j+i] = fws[nj*relidx[offset+j]+relidx[offset+i]];
	  }
	}
	
	// check if updating is necessary
	if (N2 > 0) { 
	  // extract C = F[nn:,r[N1:]]
	  for (j=0;j<N2;j++) {
	    jj = relidx[offset+N1+j];
	    for (ii=jj;ii<nj;ii++) {	      
	      ws[na*j+ii-nn] = fws[nj*jj+ii];	      
	    }
	  }

	  // reduce C to lower triangular from
	  for (j=0;j<N2;j++) {	    
	    jj = relidx[offset+N-1-j]-nn;  // index of column in C (ws)
	    ii = (na-jj)-(j+1);            // number of nonzeros to zero out
	    if (ii == 0) continue;
	    
	    // compute and apply Householder reflector
	    m = ii + 1;
	    dlarfg_(&m, ws+na*(N2-j)-1-j, ws+na*(N2-j)-1-j-ii, &iOne, &tau);
	    dtmp = ws[na*(N2-j)-1-j];
	    ws[na*(N2-j)-1-j] = 1.0;
	    i = N2-1-j;
	    dlarfx_(&cL, &m, &i, ws+na*(N2-j)-1-j-ii, &tau, ws+na-1-j-ii, &na, ws+na*na);
	    ws[na*(N2-j)-1-j] = dtmp;
	  }
			  
	  // copy lower triangular matrix from ws to 2,2 block of U
	  dlacpy_(&cL, &N2, &N2, ws+na-N2, &na, U+(N+1)*N1, &N);

	  // compute L_{21} by solving L_{22}'*L_{21} = B
	  dtrtrs_(&cL, &cT, &cN, &N2, &N1, U+N1*N+N1, &N, U+N1, &N, &info);
	  if (info) {
	    free(ws);
	    return info;
	  }
	  
	  // compute A - Li_{21}'*Li_{21}
	  dsyrk_(&cL, &cT, &N1, &N2, &dNegOne, U+N1, &N, &dOne, U, &N);
	}
	
	/* Compute Li_{11}  (reverse -- factorize -- reverse) */
	// reverse 1,1 block
	offset = (N+1)*(N1-1);
	for (j=0;j<N1/2;j++) {
	  for (i=0;i<N1;i++) {
	    dtmp = U[offset-j*N-i];
	    U[offset-j*N-i] = U[N*j+i];
	    U[N*j+i] = dtmp;
	  }
	}
	if (N1 % 2) { // nn is odd
	  j = N1/2;
	  for (i=0;i<N1/2;i++) {
	    dtmp = U[offset-j*N-i];
	    U[offset-j*N-i] = U[N*j+i];
	    U[N*j+i] = dtmp;
	  }
	}

	// factorize 1,1 block
	dpotrf_(&cU, &N1, U, &N, &info);
	if (info) {
	  free(ws);
	  return info;
	}

	// reverse 1,1 block
	offset = (N+1)*(N1-1);
	for (j=0;j<N1/2;j++) {
	  for (i=0;i<N1;i++) {
	    dtmp = U[offset-j*N-i];
	    U[offset-j*N-i] = U[N*j+i];
	    U[N*j+i] = dtmp;
	  }
	}
	if (N1 % 2) { // nn is odd
	  j = N1/2;
	  for (i=0;i<N1/2;i++) {
	    dtmp = U[offset-j*N-i];
	    U[offset-j*N-i] = U[N*j+i];
	    U[N*j+i] = dtmp;
	  }
	}
	/* end of factored update extraction */

      }
      else { /* extract unfactored update */
	for (j=0; j<N; j++) {
	  for (i=j; i<N; i++)
	    U[N*j+i] = fws[nj*relidx[offset+j]+relidx[offset+i]];
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


