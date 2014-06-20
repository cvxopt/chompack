from cvxopt import matrix, spmatrix, normal, amd, blas
from chompack import symbolic, cspmatrix, cholesky, llt, completion, projected_inverse, hessian, trsm, trmm,\
    merge_size_fill, tril, symmetrize, perm, eye
import random

def sp_rand(m,n,a):
    """
    Generates an mxn sparse 'd' matrix with round(a*m*n) nonzeros.
    """
    if m == 0 or n == 0: return spmatrix([], [], [], (m,n))
    nnz = min(max(0, int(round(a*m*n))), m*n)
    nz = matrix(random.sample(range(m*n), nnz), tc='i')
    return spmatrix(normal(nnz,1), nz%m, matrix([int(ii) for ii in nz/m]), (m,n))

random.seed(1)
# Generate random sparse matrix of order ...
n = 200
# and with density ...
rho = 0.02

As = sp_rand(n,n,rho) + spmatrix(10.0,range(n),range(n))

# We will use (greedy) clique merging in this example:
fmerge = merge_size_fill(16,4)

# Compute symbolic factorization with AMD ordering and clique merging; only lower triangular part of As is accessed
print("Computing symbolic factorization..")
p = amd.order
symb = symbolic(As, p = p, merge_function = fmerge)

print("Order of matrix       : %i" % (symb.n))
print("Number of nonzeros    : %i" % (symb.nnz))
print("Number of supernodes  : %i" % (symb.Nsn))
print("Largest supernode     : %i" % (max([symb.snptr[k+1]-symb.snptr[k] for k in range(symb.Nsn)])))
print("Largest clique        : %i\n" % (symb.clique_number))

A = cspmatrix(symb)  # create new cspmatrix from symbolic factorization
A += As              # add spmatrix 'As' to cspmatrix 'A'; this ignores the upper triangular entries in As

print("Computing Cholesky factorization..")
L = A.copy()         # make a copy of A
cholesky(L)          # compute Cholesky factorization; overwrites L

print("Computing Cholesky product..")
At = L.copy()        # make a copy of L
llt(At)              # compute Cholesky product; overwrites At

print("Computing projected inverse..")
Y = L.copy()         # make a copy of L
projected_inverse(Y) # compute projected inverse; overwrites Y

print("Computing completion..")
Lc = Y.copy()        # make a copy of Y
completion(Lc, factored_updates = False) # compute completion; overwrites Lc

print("Computing completion with factored updates..")
Lc2 = Y.copy()       # make a copy of Y
completion(Lc2, factored_updates = True) # compute completion (with factored updates); overwrites Lc2

print("Applying Hessian factors..")
U = At.copy()
fupd = False
hessian(L, Y, U, adj = False, inv = False, factored_updates = fupd)
hessian(L, Y, U, adj = True, inv = False, factored_updates = fupd)
hessian(L, Y, U, adj = True, inv = True, factored_updates = fupd)
hessian(L, Y, U, adj = False, inv = True, factored_updates = fupd)

print("\nEvaluating errors:\n")
# Compute norm of error: A - L*L.T
tmp = (A-At).spmatrix().V
print("Cholesky factorization/product     :  err = %.3e" % (blas.nrm2(tmp)))

# Compute norm of error: L - Lc
tmp = (L.spmatrix()-Lc.spmatrix()).V
print("Projected inverse/completion       :  err = %.3e" % (blas.nrm2(tmp)))

# Compute norm of error: L - Lc2
tmp = (L.spmatrix()-Lc2.spmatrix()).V
print("Projected inverse/completion (upd) :  err = %.3e" % (blas.nrm2(tmp)))

# Compute norm of error: At - U
tmp = (At-U).spmatrix().V
print("Hessian factors NN/TN/TI/NI        :  err = %.3e" % (blas.nrm2(tmp)))


# Test triangular matrix products and solve
p = L.symb.p

B = eye(n)
trsm(L, B)
print("trsm, trans = 'N'                  :  err = %.3e" % (blas.nrm2(L.spmatrix(reordered = True)*B[p,p] - eye(n))))

B = eye(n)
trsm(L, B, trans = 'T')
print("trsm, trans = 'T'                  :  err = %.3e" % (blas.nrm2(L.spmatrix(reordered = True).T*B[p,p] - eye(n))))

B = eye(n)
trmm(L, B)
print("trmm, trans = 'N'                  :  err = %.3e" % (blas.nrm2(L.spmatrix(reordered = True) - B[p,p])))

B = eye(n)
trmm(L, B, trans = 'T')
print("trmm, trans = 'T'                  :  err = %.3e" % (blas.nrm2(L.spmatrix(reordered = True).T - B[p,p])))

B = eye(n)
trmm(L,B,trans='T')
trmm(L,B)
print("llt(L) - trmm N/T                  :  err = %.3e" % (blas.nrm2(tril(B - As))))
