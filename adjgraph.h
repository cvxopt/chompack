#ifndef __ADJGRAPH__
#define __ADJGRAPH__

#include "chtypes.h"

typedef struct vertex {

  int value;
  struct vertex *prev, *next;

} vertex;

typedef struct vertexlist {

  vertex *head, *tail;
  int size;

} vertexlist;

typedef struct {

  vertexlist **adjlist;
  int numVertices;

} adjgraph;

/* prototypes */
adjgraph * adjgraph_create(const ccs *);
void adjgraph_destroy(adjgraph *);
adjgraph * adjgraph_copy(const adjgraph *);
int adjgraph_symmetrize(adjgraph *);
adjgraph * adjgraph_create_symmetric(const ccs *X);
int triangulate(adjgraph *A);

vertexlist * vertexlist_new(void );
void vertexlist_destroy(vertexlist *);
int vertexlist_insert_before(vertexlist *, vertex *, int);
int vertexlist_insert_after(vertexlist *, vertex *, int);
void vertexlist_remove(vertexlist *, vertex *);
int vertexlist_is_member(const vertexlist *, int);

int maxcardsearch(const adjgraph *A, int_t *order);
int is_peo(const adjgraph *A, const int_t *order, const int_t *inv_order);

#endif
