#include "chompack.h"

void _Y2K(const int_t n,   // order of matrix
	  const int_t nsn,       // number of supernodes/cliques
	  const int_t *snpost,   // post-ordering of supernodes
	  const int_t *snptr,    // supernode pointer
	  const int_t *relptr,
	  const int_t *relidx,
	  const int_t *chptr,
	  const int_t *chidx,
	  const int_t *blkptr,
	  double * restrict lblkval,
	  double *restrict *restrict ublkval,
	  double * restrict fws,  // frontal matrix workspace
	  double * restrict upd,  // update matrix workspace
	  int_t * restrict upd_size,
	  int inv) {

  int nn,na,nj,offset,i,j,k,ki,l,N,nup=0,uk=0;
  double * restrict U, * restrict ublkvalk;
  double dOne=1.0,alpha=-1.0;
  char cL='L',cT='T',cR='R',cN='N';

  if (inv) alpha = 1.0;

  U = upd;   // pointer to top of update storage

  while ((ublkvalk = ublkval[uk++])) {

    for (ki=0;ki<nsn;ki++) {
      k = snpost[ki];
      nn = snptr[k+1]-snptr[k];
      na = relptr[k+1]-relptr[k];
      nj = na + nn;

      // copy Ut_{Jk,Nk} to leading columns of fws
      dlacpy_(&cL, &nj, &nn, ublkvalk+blkptr[k], &nj, fws, &nj);
      for (j=nn;j<nj;j++) {
	for (i=j;i<nj;i++) {
	  fws[nj*j+i] = 0.0; // zero out (2,2) block of frontal matrix
	}
      }

      if (!inv) {
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
      }

      if (na > 0) {
	// F_{Ak,Ak} := F_{Ak,Ak} + alpha*L_{Ak,Nk}*F_{Ak,Nk}'
	dgemm_(&cN, &cT, &na, &na, &nn, &alpha, lblkval+blkptr[k]+nn, &nj,
	       fws+nn, &nj, &dOne, fws+(nj+1)*nn, &nj);
	// F_{Ak,Nk} := F_{Ak,Nk} + alpha*L_{Ak,Nk}*F_{Nk,Nk}
	dsymm_(&cR, &cL, &na, &nn, &alpha, fws, &nj, lblkval+blkptr[k]+nn, &nj, &dOne, fws+nn, &nj);
	// F_{Ak,Ak} := F_{Ak,Ak} + alpha*F_{Ak,Nk}*L_{Ak,Nk}'
	dgemm_(&cN, &cT, &na, &na, &nn, &alpha, fws+nn, &nj, lblkval+blkptr[k]+nn, &nj, &dOne, fws+(nj+1)*nn, &nj);
      }

      if (inv) {
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
      }

      if (na > 0) {
	// copy update matrix to stack
	upd_size[nup++] = na;
	dlacpy_(&cL, &na, &na, fws+nn*nj+nn, &nj, U, &na);
	U += na*na;
      }

      // copy the leading nn columns of frontal matrix to Ut
      dlacpy_(&cL, &nj, &nn, fws, &nj, ublkvalk+blkptr[k], &nj);

    }
  }

  return;
}

void _M2T(const int_t n,         // order of matrix
	  const int_t nsn,       // number of supernodes/cliques
	  const int_t *snpost,   // post-ordering of supernodes
	  const int_t *snptr,    // supernode pointer
	  const int_t *relptr,
	  const int_t *relidx,
	  const int_t *chptr,
	  const int_t *chidx,
	  const int_t *blkptr,
	  double * restrict lblkval,
	  double *restrict *restrict ublkval,
	  double * restrict fws,  // frontal matrix workspace
	  double * restrict upd,  // update matrix workspace
	  int_t * restrict upd_size,
	  int inv) {

  int nn,na,nj,offset,i,j,k,ki,l,N,nup=0,uk=0;
  double * restrict U, * restrict ublkvalk;
  double dOne=1.0,alpha=-1.0;
  char cL='L',cT='T',cN='N';

  if (inv) alpha = 1.0;

  U = upd;   // pointer to top of update storage

  while ((ublkvalk = ublkval[uk++])) {

    for (ki=nsn-1;ki>=0;ki--) {
      k = snpost[ki];
      nn = snptr[k+1]-snptr[k];
      na = relptr[k+1]-relptr[k];
      nj = na + nn;

      // copy Ut_{Jk,Nk} to leading columns of F
      dlacpy_(&cL, &nj, &nn, ublkvalk+blkptr[k], &nj, fws, &nj);

      // if supernode k is not a root node:
      if (na > 0) {
	// copy update matrix to 2,2 block of frontal matrix
	nup--;
	U -= upd_size[nup]*upd_size[nup];
	dlacpy_(&cL, &na, &na, U, &na, fws+(nj+1)*nn, &nj);
      }

      /*
	Compute T_{Jk,Nk} (stored in leading columns of fws)
      */

      if (inv) {
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
      }

      // if supernode k is not a root node:
      if (na > 0) {
	// F_{Nk,Nk} := F_{Nk,Nk} + alpha*F_{Ak,Nk}'*L_{Ak,Nk}
	dgemm_(&cT, &cN, &nn, &nn, &na, &alpha, fws+nn, &nj, lblkval+blkptr[k]+nn, &nj, &dOne, fws, &nj);
	// F_{Ak,Nk} := F_{Ak,Nk} + alpha*F_{Ak,Ak}*L_{Ak,Nk}
	dsymm_(&cL, &cL, &na, &nn, &alpha, fws+(nj+1)*nn, &nj, lblkval+blkptr[k]+nn, &nj, &dOne, fws+nn, &nj);
	// F_{Nk,Nk} := F_{Nk,Nk} + alpha*L_{Ak,Nk}'*F_{Ak,Nk}
	dgemm_(&cT, &cN, &nn, &nn, &na, &alpha, lblkval+blkptr[k]+nn, &nj, fws+nn, &nj, &dOne, fws, &nj);
      }

      // copy the leading nn columns of frontal matrix to Ut
      dlacpy_(&cL, &nj, &nn, fws, &nj, ublkvalk+blkptr[k], &nj);

      if (!inv) {
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
      }

    }
  }

  return;
}


int _scale(const int_t n,         // order of matrix
	   const int_t nsn,       // number of supernodes/cliques
	   const int_t *snpost,   // post-ordering of supernodes
	   const int_t *snptr,    // supernode pointer
	   const int_t *relptr,
	   const int_t *relidx,
	   const int_t *chptr,
	   const int_t *chidx,
	   const int_t *blkptr,
	   double * restrict lblkval,
	   double * restrict yblkval,
	   double *restrict *restrict ublkval,
	   double * restrict fws,  // frontal matrix workspace
	   double * restrict upd,  // update matrix workspace
	   int_t * restrict upd_size,
	   int inv,
	   int adj,
	   int factored_updates) {

  int nn,na,nj,offset,info,i,j,k,ki,l,N,nup=0,uk=0;
  double * restrict U, * restrict ublkvalk, * restrict ws=NULL;
  double dOne=1.0,alpha=-1.0;
  char cL='L',cT='T',cR='R',cN='N';
  char *tr1, *tr2, *tr3=NULL;

  if (inv) alpha = 1.0;

  U = upd;   // pointer to top of update storage

  for (ki=nsn-1;ki>=0;ki--) {
    k = snpost[ki];
    nn = snptr[k+1]-snptr[k];
    na = relptr[k+1]-relptr[k];
    nj = na + nn;

    // copy Y_{Jk,Nk} to leading columns of F
    dlacpy_(&cL, &nj, &nn, yblkval+blkptr[k], &nj, fws, &nj);
    for (j=nn;j<nj;j++) {
      for (i=j;i<nj;i++) {
	fws[nj*j+i] = 0.0; // zero out (2,2) block of frontal matrix
      }
    }

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
    if (na > 0) {
      if (factored_updates) {
	// In this case we have Vk = Lk'*Lk
	if (adj) tr3 = &cT;
	else tr3 = &cN;
      }
      else {
	// factorize Vk
	dpotrf_(&cL, &na, fws+(nj+1)*nn, &nj, &info);
	if (info) return info;
	// In this case we have Vk = Lk*Lk'
	if (adj) tr3 = &cN;
	else tr3 = &cT;
      }
    }

    if (adj) {
      tr1 = &cN;
      tr2 = &cT;
    }
    else {
      tr1 = &cT;
      tr2 = &cN;
    }

    uk = 0;
    while ((ublkvalk = ublkval[uk++])) {
      // symmetrize (1,1) block of U[k]
      for (j=0;j<nn-1;j++) {
	for (i=j+1;i<nn;i++) {
	  ublkvalk[blkptr[k]+i*nj+j] = ublkvalk[blkptr[k]+nj*j+i];
	}
      }

      // apply scaling
      if (!inv) {
	dtrsm_(&cR, &cL, tr1, &cN, &nj, &nn, &dOne, lblkval+blkptr[k], &nj, ublkvalk+blkptr[k], &nj);
	dtrsm_(&cL, &cL, tr2, &cN, &nn, &nn, &dOne, lblkval+blkptr[k], &nj, ublkvalk+blkptr[k], &nj);
	// zero-out strict upper triangular part of {Nj,Nj} block 
	for (j=1;j<nn;j++) {
	  for (i=0;i<j;i++) ublkvalk[blkptr[k]+j*nj+i] = 0.0;
	}
	if (na > 0) {
	  dtrmm_(&cL, &cL, tr3, &cN, &na, &nn, &dOne, fws+(nj+1)*nn, &nj, ublkvalk+blkptr[k]+nn, &nj);
	}
      }
      else {
	dtrmm_(&cR, &cL, tr1, &cN, &nj, &nn, &dOne, lblkval+blkptr[k], &nj, ublkvalk+blkptr[k], &nj);
	dtrmm_(&cL, &cL, tr2, &cN, &nn, &nn, &dOne, lblkval+blkptr[k], &nj, ublkvalk+blkptr[k], &nj);
	// zero-out strict upper triangular part of {Nj,Nj} block 
	for (j=1;j<nn;j++) {
	  for (i=0;i<j;i++) ublkvalk[blkptr[k]+j*nj+i] = 0.0;
	}
	if (na > 0) {
	  dtrsm_(&cL, &cL, tr3, &cN, &na, &nn, &dOne, fws+(nj+1)*nn, &nj, ublkvalk+blkptr[k]+nn, &nj);
	}
      }
    }
  }

  return 0;
}

int hessian(const int_t n,        
	    const int_t nsn,      
	    const int_t *snpost,  
	    const int_t *snptr,   
	    const int_t *relptr,
	    const int_t *relidx,
	    const int_t *chptr,
	    const int_t *chidx,
	    const int_t *blkptr,
	    double * restrict lblkval,
	    double * restrict yblkval,
	    double *restrict *restrict ublkval,
	    double * restrict fws,
	    double * restrict upd,
	    int_t * restrict upd_size,
	    int inv,
	    int adj,
	    int factored_updates) {

  if (adj != inv) _scale(n,nsn,snpost,snptr,relptr,relidx,chptr,chidx,blkptr,lblkval,yblkval,ublkval,fws,upd,upd_size,inv,adj,factored_updates);
  if (adj) _M2T(n,nsn,snpost,snptr,relptr,relidx,chptr,chidx,blkptr,lblkval,ublkval,fws,upd,upd_size,inv);
  else     _Y2K(n,nsn,snpost,snptr,relptr,relidx,chptr,chidx,blkptr,lblkval,ublkval,fws,upd,upd_size,inv);
  if (adj == inv) _scale(n,nsn,snpost,snptr,relptr,relidx,chptr,chidx,blkptr,lblkval,yblkval,ublkval,fws,upd,upd_size,inv,adj,factored_updates);

  return 0;
}
