#ifndef __CHOLUPDATE__
#define __CHOLUPDATE__

int chol_add_row(double *L, int n, int ld, double *a, int k, double *work);
int chol_delete_row(double *L, int n, int ld, int k, double *work);
int chol_rank1_update(double *L, int n, int ld, double a, double *z, double *work);

#endif
