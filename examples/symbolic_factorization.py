from cvxopt import spmatrix, printing, amd
import chompack as cp
printing.options['width'] = 17

# Define sparse matrix
I = range(17) + [2,2,3,3,4,14,4,14,8,14,15,8,15,7,8,14,8,14,14,15,10,12,13,16,12,13,16,12,13,15,16,13,15,16,15,16,15,16,16]
J = range(17) + [0,1,1,2,2,2,3,3,4,4,4,5,5,6,6,6,7,7,8,8,9,9,9,9,10,10,10,11,11,11,11,12,12,12,13,13,14,14,15]
A = spmatrix(1.0,I,J,(17,17))

# Test if A is chordal
p = cp.maxcardsearch(A)
print("\nMaximum cardinality search")
print(" -- perfect elimination order:"), cp.peo(A,p)

# Test if natural ordering 0,1,2,...,17 is a perfect elimination order
p = range(17)
print("\nNatural ordering")
print(" -- perfect elimination order:"), cp.peo(A,p)

p = amd.order(A)
print("\nAMD ordering")
print(" -- perfect elimination order:"), cp.peo(A,p)

# Compute a symbolic factorization 
symb = cp.symbolic(A, p)
print("\nSymbolic factorization:")
print("Fill              :"), sum(symb.fill)
print("Number of cliques :"), symb.Nsn
print(symb)

# Compute a symbolic factorization with clique merging
symb2 = cp.symbolic(A, p, merge_function = cp.merge_size_fill(3,3))
print("Symbolic factorization with clique merging:")
print("Fill (fact.+merging) :"), sum(symb2.fill)
print("Number of cliques    :"), symb2.Nsn
print(symb2)
