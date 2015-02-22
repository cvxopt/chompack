from cvxopt import matrix, spmatrix, sparse, normal, solvers
import chompack as cp
import random

# generate random sparse matrix
def sp_rand(m,n,a):
    """
    Generates an m-by-n sparse 'd' matrix with round(a*m*n) nonzeros.
    """
    if m == 0 or n == 0: return spmatrix([], [], [], (m,n))
    nnz = min(max(0, int(round(a*m*n))), m*n)
    nz = matrix(random.sample(range(m*n), nnz), tc='i')
    return spmatrix(normal(nnz,1), nz%m, nz/m, (m,n))

# generate random sparsity pattern and sparse SDP problem data
random.seed(1)
m, n = 50, 200
print("Generating random sparse SDP (n=%i, m=%i constraints).."%(n,m))
A = sp_rand(n,n,0.015) + spmatrix(1.0,range(n),range(n))
I = cp.tril(A)[:].I
N = len(I)/50 # each data matrix has 1/50 of total nonzeros in pattern
Ig = []; Jg = []
for j in range(m):
    Ig += sorted(random.sample(I,N))   
    Jg += N*[j]
G = spmatrix(normal(len(Ig),1),Ig,Jg,(n**2,m))
h = G*normal(m,1) + spmatrix(1.0,range(n),range(n))[:]
c = normal(m,1)
dims =  {'l':0, 'q':[], 's': [n]};

# solve SDP with CVXOPT 
print("Solving SDP with CVXOPT..")
prob = (c, G, matrix(h), dims)
sol = solvers.conelp(*prob)
Z1 = matrix(sol['z'], (n,n))

# convert SDP and solve
prob2, blocks_to_sparse, symbs = cp.convert_conelp(*prob)
print("Solving converted SDP (no merging)..")
sol2 = solvers.conelp(*prob2) 

# convert block-diagonal solution to spmatrix
blki,I,J,bn = blocks_to_sparse[0]
Z2 = spmatrix(sol2['z'][blki],I,J)

# compute completion
symb = cp.symbolic(Z2, p=cp.maxcardsearch)
Z2c = cp.psdcompletion(cp.cspmatrix(symb)+Z2, reordered=False)


prob3, blocks_to_sparse, symbs = cp.convert_conelp(*prob, coupling = 'full', merge_function = cp.merge_size_fill(5,5))
print("Solving converted SDP (with merging)..")
sol3 = solvers.conelp(*prob3) 

# convert block-diagonal solution to spmatrix
blki,I,J,bn = blocks_to_sparse[0]
Z3 = spmatrix(sol3['z'][blki],I,J)

# compute completion
symb = cp.symbolic(Z3, p=cp.maxcardsearch)
Z3c = cp.psdcompletion(cp.cspmatrix(symb)+Z3, reordered=False)
