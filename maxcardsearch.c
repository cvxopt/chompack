#include <stdio.h>
#include <stdlib.h>

#include "adjgraph.h"

int maxcardsearch(const adjgraph *B, int_t *order) {

  int i, n, v, maxSetIdx = 0;
  int *labeled, *Sidx;
  vertexlist **S;
  vertex **Sptr, *p;

  adjgraph *A = adjgraph_copy(B);
  if (!A) return -1;

  n = A->numVertices;

  labeled = calloc(n, sizeof(int));
  Sidx = calloc(n, sizeof(int));     /* S index of vertices */
  S = calloc(n, sizeof(vertexlist *)); /* S sets */
  Sptr = malloc(n*sizeof(vertex *));   /* ptr from vertices into S*/

  if (!labeled || !Sidx || !S || !Sptr) goto error_cleanup;

  for (i=0; i<n; i++)
    if (!(S[i] = vertexlist_new())) goto error_cleanup;

  for (i=0; i<n; i++) {
    if (vertexlist_insert_after(S[0], S[0]->tail, i)) goto error_cleanup;
    Sptr[i] = S[0]->tail;
  }

  for (i=0; i<n; i++) {

#ifdef DEBUG
    printf("ITERATION %i\n",i+1);
    int k;
    for (k=0; k<n; k++) {
      printf("S[%i] :",k);
      vertexlist_print(S[k]);
    }
#endif

    v = S[maxSetIdx]->tail->value;

#ifdef DEBUG
    printf("picked vertex %i from S[%i]\n", v, maxSetIdx);
#endif

    vertexlist_remove(S[maxSetIdx], S[maxSetIdx]->tail);

    labeled[v] = 1;
    order[n-1-i] = v;

    while (i < n-1 && !S[maxSetIdx]->size) maxSetIdx--;

    p = A->adjlist[v]->tail;
    while (p) {

      if (!labeled[p->value]) {

#ifdef DEBUG
        printf("moving %i from S[%i] to S[%i]\n",
            p->value, Sidx[p->value], Sidx[p->value]+1);
#endif

        /* move vertex up one level in S */
        vertexlist_remove(S[ Sidx[p->value] ], Sptr[p->value] );

        Sidx[ p->value ]++;

        vertexlist_insert_after(S[ Sidx[p->value] ],
            S[ Sidx[p->value] ]->tail, p->value );

        Sptr[ p->value ] = S[ Sidx[p->value] ]->tail;

        if (Sidx[p->value] > maxSetIdx)
          maxSetIdx = Sidx[p->value];
      }

      /* remove vertex from adjacency set*/
      vertexlist_remove(A->adjlist[v], A->adjlist[v]->tail);
      p = A->adjlist[v]->tail;
    }
  }

  free(Sidx);
  free(labeled);

  for (i=0; i<n; i++)
    vertexlist_destroy(S[i]);

  free(S);
  free(Sptr);
  adjgraph_destroy(A);

  return 0;

  error_cleanup:

  free(Sidx);
  free(labeled);

  for (i=0; i<n; i++)
    vertexlist_destroy(S[i]);

  free(S);
  free(Sptr);
  adjgraph_destroy(A);

  return -1;
}


int is_peo(const adjgraph *A, const int_t *order, const int_t *inv_order) {

  int i, n = A->numVertices;
  int u, v;

#ifdef DEBUG
  adjgraph_print(A);
#endif

  for (i=0; i<n; i++) {

    vertex *p;
    v = order[i];
#ifdef DEBUG
    printf("v = %i\n", v);
#endif

    /* find the neighbor to be eliminated next */

    u = -1;
    p = A->adjlist[v]->head;
    while (p) {
      if ((inv_order[p->value] > inv_order[v]) &&
          (u<0 || inv_order[u] > inv_order[p->value]))
        u = p->value;

      p = p->next;
    }

#ifdef DEBUG
    if (u>=0)
      printf("next neighbor to eliminate: %i\n", u);
    else {
      printf("No neighbor left to eliminate\n");
    }
#endif

    /*
       If there is a node in madj(v) not adjacent to u, then
       the elimination ordering is not perfect
     */
    if (u >= 0) {
      p = A->adjlist[v]->tail;
      while (p) {
        if ((inv_order[p->value] > inv_order[u]) &&
            !vertexlist_is_member(A->adjlist[p->value], u)) {

#ifdef DEBUG
              printf("%i is not adjacent to %i\n", p->value, u);

              printf("adj(%i): ", p->value);
              vertexlist_print(A->adjlist[p->value]);
#endif
              return 0;
        }

        p = p->prev;
      }
    }

  }

  return 1;
}
