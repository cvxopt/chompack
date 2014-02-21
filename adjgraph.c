#include <stdio.h>
#include <stdlib.h>

#if !defined(__APPLE__)
#include <malloc.h>
#endif


#include "adjgraph.h"

vertexlist * vertexlist_new() {

  vertexlist *L = malloc(sizeof(vertexlist));
  L->head = NULL;
  L->tail = NULL;
  L->size = 0;

  return L;
}

void vertexlist_destroy(vertexlist *L) {

  if (L) {
    vertex *p = L->head, *pnext;

    while (p) {
      pnext = p->next;
      free(p);
      p = pnext;
    }

    free(L);
  }
}

int vertexlist_insert_before(vertexlist *L, vertex *v, int value) {

  vertex *u = malloc(sizeof(vertex));
  if (!u) return -1;
  u->value = value;

  if (!v) {
    L->head = u;
    L->tail = u;
    u->next = NULL;
    u->prev = NULL;
  }
  else {

    u->next = v;
    u->prev = v->prev;
    v->prev = u;
    if (u->prev) u->prev->next = u;
    if (L->head == v) L->head = u;
  }
  L->size++;

  return 0;
}

int vertexlist_insert_after(vertexlist *L, vertex *v, int value) {

  vertex *u = malloc(sizeof(vertex));
  if (!u) return -1;
  u->value = value;

  if (!v) {
    L->head = u;
    L->tail = u;
    u->next = NULL;
    u->prev = NULL;
  }
  else {
    u->next = v->next;
    v->next = u;
    u->prev = v;
    if (u->next) u->next->prev = u;
    if (L->tail == v) L->tail = u;
  }
  L->size++;

  return 0;
}

void vertexlist_remove(vertexlist *L, vertex *v) {

  if ( v->prev )
    v->prev->next = v->next;
  else
    L->head = v->next;

  if ( v->next )
    v->next->prev = v->prev;
  else
    L->tail = v->prev;

  free(v);
  L->size--;
}

int vertexlist_is_member(const vertexlist *L, int value) {

  vertex *v = L->tail;
  while (v) {
    if (v->value == value) return 1;
    v = v->prev;
  }

  return 0;
}

#ifdef DEBUG
void vertexlist_print(const vertexlist *L) {

  vertex *p = L->head;

  while (p) {
    printf("%i ",p->value);
    p = p->next;
  }
  printf("\n");
}

void adjgraph_print(const adjgraph *A) {

  printf("PRINTING ADJGRAPH\n");
  int i;
  for (i=0; i<A->numVertices; i++) {
    printf("VERTEX %i: ",i);
    vertexlist_print(A->adjlist[i]);
  }
}
#endif

void adjgraph_destroy(adjgraph *A) {

  int i;
  for (i=0; i<A->numVertices; i++)
    vertexlist_destroy(A->adjlist[i]);

  free(A->adjlist);
  free(A);
}

adjgraph * adjgraph_create(const ccs *X) {

  int i, k;
  int_t n = X->nrows, *colptr = X->colptr, *rowind = X->rowind;

  adjgraph *A = malloc(sizeof(adjgraph));
  if (!A) return NULL;
  A->numVertices = n;
  if (!(A->adjlist = calloc(n, sizeof(vertexlist *)))) {
    free(A);
    return NULL;
  }

  for (i=0; i<n; i++) {
    if (!(A->adjlist[i] = vertexlist_new())) {
      adjgraph_destroy(A);
      return NULL;
    }
  }

  for (i=0; i<n; i++) {
    for (k=colptr[i]; k<colptr[i+1]; k++) {
      if (rowind[k] < i) continue;

      if (rowind[k] != i) {

        if (vertexlist_insert_after(A->adjlist[i],
            A->adjlist[i]->tail, rowind[k])) {
              adjgraph_destroy(A);
              return NULL;
        }
      }
    }
  }

  return A;
}

adjgraph * adjgraph_copy(const adjgraph *B) {

  int i;
  adjgraph *A = malloc(sizeof(adjgraph));
  if (!A) return NULL;
  A->numVertices = B->numVertices;

  if (!(A->adjlist = calloc(A->numVertices, sizeof(vertexlist *)))) {
    free(A);
    return NULL;
  }

  for (i=0; i<A->numVertices; i++) {
    if (!(A->adjlist[i] = vertexlist_new())) {
      adjgraph_destroy(A);
      return NULL;
    }
  }

  for (i=0; i<A->numVertices; i++) {
    vertex *p = B->adjlist[i]->head;
    while (p) {
      if (vertexlist_insert_after(A->adjlist[i],
          A->adjlist[i]->tail, p->value)) {
            adjgraph_destroy(A);
            return NULL;
      }

      p = p->next;
    }
  }

  return A;
}

int triangulate(adjgraph *A) {

  int i, k;
  vertex *pi, *pk;

  for (i=0; i<A->numVertices; i++) {

    if (A->adjlist[i]->size > 1) {
      k = A->adjlist[i]->head->value;

      /* merge adjlist[i]\{k} with adjlist[k] */
      pi = A->adjlist[i]->head->next;
      pk = (A->adjlist[k]->size ? A->adjlist[k]->head : NULL);

      while (pi) {

        if (pk && pi->value == pk->value) {

          pi = pi->next; pk = pk->next;

        }
        else if (pk && pi->value > pk->value) {

          pk = pk->next;

        }
        else if (pk && pi->value < pk->value) {

          if (vertexlist_insert_before(A->adjlist[k], pk, pi->value))
            return -1;

          pi = pi->next;

        }
        else {

          if (pi->value == k) break;  /* we don't store diagonal elmns */

          if (vertexlist_insert_after(A->adjlist[k],
              A->adjlist[k]->tail, pi->value))
            return -1;

          pi = pi->next;
        }
      }
    }
  }

  return 0;
}

int adjgraph_symmetrize(adjgraph *A)
{
  int i;
  for (i=0; i<A->numVertices; i++) {

    vertex *p = A->adjlist[i]->head;
    while(p) {

      if (p->value > i && vertexlist_insert_after(A->adjlist[ p->value ],
          A->adjlist[ p->value ]->tail, i))
        return -1;

      p = p->next;
    }
  }

  return 0;
}

adjgraph * adjgraph_create_symmetric(const ccs *X) {

  adjgraph *A = adjgraph_create(X);
  if (A && adjgraph_symmetrize(A)) {
    adjgraph_destroy(A);
    return NULL;
  }

  return A;
}

