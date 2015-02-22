#include "chompack.h"

void trsm(const char trans, 
	  int nrhs,
	  const double alpha,
	  const int_t n,         // order of matrix
	  const int_t nsn,       // number of supernodes/cliques
	  const int_t *snpost,   // post-ordering of supernodes
	  const int_t *snptr,    // supernode pointer
	  const int_t *snode,    // supernode array
	  const int_t *relptr,
	  const int_t *relidx,
	  const int_t *chptr,
	  const int_t *chidx,
	  const int_t *blkptr,
	  const int_t *p,
	  double * restrict blkval,
	  double * restrict a,
	  int * lda,
	  double * restrict fws,  // frontal matrix workspace
	  double * restrict upd,  // update matrix workspace
	  int_t * restrict upd_size
	  ) {

  int nn,na,nj,offset,i,j,k,ki,ir,l,N,nup=0;
  double * restrict U;
  double dOne=1.0,dNegOne=-1.0;
  char cL = 'L', cT = 'T', cN = 'N';

  U = upd;   // pointer to top of update storage

  if (trans == 'N') {
    for (ki=0;ki<nsn;ki++) {
      k = snpost[ki];
      nn = snptr[k+1]-snptr[k];
      na = relptr[k+1]-relptr[k];
      nj = na + nn;

      // extract block from rhs
      for (j=0;j<nrhs;j++) {
	offset = nj*j;
	for (i=0;i<nn;i++) {
	  ir = snode[snptr[k]+i];
	  fws[offset+i] = alpha*a[j*(*lda)+p[ir]];
	}
	for (i=nn;i<nj;i++) {
	  fws[offset+i] = 0.0;
	}
      }

      // add contributions from children
      for (l=chptr[k+1]-1;l>=chptr[k];l--) {
	nup--;
	U -= upd_size[nup]*nrhs;	
	offset = relptr[chidx[l]];
	N = relptr[chidx[l]+1] - offset;
	for (j=0;j<nrhs;j++) {
	  for (i=0; i<N; i++) {
	    fws[nj*j+relidx[offset+i]] += U[N*j+i]; 
	  }
	}
      }

      // if k is not a root node
      if (na > 0) {
	dgemm_(&cN,&cN,&na,&nrhs,&nn,&dNegOne,blkval+blkptr[k]+nn,&nj,fws,&nj,&dOne,fws+nn, &nj);
	upd_size[nup++] = na;
	dlacpy_(&cN, &na, &nrhs, fws+nn, &nj, U, &na); 
	U += na*nrhs;
      }

      // scale and copy block to rhs
      dtrsm_(&cL, &cL, &cN, &cN, &nn, &nrhs, &dOne, blkval+blkptr[k], &nj, fws, &nj);
      for (j=0;j<nrhs;j++) {
	offset = nj*j;
	for (i=0;i<nn;i++) {
	  ir = snode[snptr[k]+i];
	  a[j*(*lda) + p[ir]] = fws[offset+i];
	}
      }
     
    }
  }
  else if (trans == 'T') {
    
    for (ki=nsn-1;ki>=0;ki--) {
      k = snpost[ki];
      nn = snptr[k+1]-snptr[k];
      na = relptr[k+1]-relptr[k];
      nj = na + nn;

      // extract block from rhs
      for (j=0;j<nrhs;j++) {
	offset = nj*j;
	for (i=0;i<nn;i++) {
	  ir = snode[snptr[k]+i];
	  fws[offset+i] = alpha*a[j*(*lda)+p[ir]];
	}
	for (i=nn;i<nj;i++) {
	  fws[offset+i] = 0.0;
	}
      }
      dtrsm_(&cL, &cL, &cT, &cN, &nn, &nrhs, &dOne, blkval+blkptr[k], &nj, fws, &nj);
      
      // if k is not a root node
      if (na > 0) {
	nup--;
	U -= upd_size[nup]*nrhs;
	dlacpy_(&cN, &na, &nrhs, U, &na, fws+nn, &nj); 
	dgemm_(&cT,&cN,&nn,&nrhs,&na,&dNegOne,blkval+blkptr[k]+nn,&nj,fws+nn,&nj,&dOne,fws,&nj);
      }

      // stack contributions for children
      for (l=chptr[k];l<chptr[k+1];l++) {
	offset = relptr[chidx[l]];
	N = relptr[chidx[l]+1]-offset;
	upd_size[nup++] = N;
	for (j=0;j<nrhs;j++) {
	  for (i=0; i<N; i++) {
	    U[N*j+i] = fws[nj*j+relidx[offset+i]];
	  }
	}
	U += N*nrhs;
      }
      
      // copy block to rhs
      for (j=0;j<nrhs;j++) {
	offset = nj*j;
	for (i=0;i<nn;i++) {
	  ir = snode[snptr[k]+i];
	  a[j*(*lda) + p[ir]] = fws[offset+i];
	}
      }
            
    }
  }
  return;
}

