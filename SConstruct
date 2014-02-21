env = Environment()

SRCS = 'adjgraph.c chompack.c cliquetree.c maxcardsearch.c sparse.c cholupdate.c pf_cholesky.c maxchord.c'

import sys
if sys.maxint > 2**31: env.Append(CPPDEFINES = ['DLONG'])
	
#env.Append(LIBPATH = ['/usr/lib'])

omp = ARGUMENTS.get('OMP', 0)
if int(omp):
	env.Append(CCFLAGS = '-fopenmp', LIBS = ['gomp'])

chompack = env.SharedLibrary( 'chompack', Split(SRCS), LIBS = ['lapack','blas','amd'])
env.Program('example', 'example.c', LIBS = [chompack])