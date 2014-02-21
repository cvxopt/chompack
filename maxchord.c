#include <stdio.h>
#include <stdlib.h>
#include "adjgraph.h"

#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

void vertexlist_pop(vertexlist *L, vertex *v) {
  if ( v->prev )
    v->prev->next = v->next;
  else
    L->head = v->next;
  if ( v->next )
    v->next->prev = v->prev;
  else
    L->tail = v->prev;
  L->size--;
  return;
}

void vertexlist_push(vertexlist *L, vertex *v) {
  int i;
  vertex *u;

  if (!v) return;
  if (L->size == 0) {
    L->head = v;
    L->tail = v;
    v->prev = NULL;
    v->next = NULL;
  }
  else {
    u = L->head;
    for (i=0;i<L->size;i++) {
      if (u->value > v->value) break;
      u = u->next;
    }
    if (u == NULL) {
      // insert after tail
      L->tail->next = v;
      v->prev = L->tail;
      v->next = NULL;
      L->tail = v;
    }
    else {
      // insert before u
      v->next = u;
      v->prev = u->prev;
      u->prev = v;
      if (u == L->head) L->head = v;
      else v->prev->next = v;
    }
  }
  L->size++;
  return;
}

int vertexlist_insert(vertexlist *L, int value) {
  vertex *u = malloc(sizeof(vertex));
  if (!u) return -1;
  u->value = value;
  vertexlist_push(L,u);
  return 0;
}

adjgraph *maxchord(const adjgraph *A, const int_t startvertex, int_t **order) {

  int_t i, j, k, l, n, degree_max = 0, compl = 0;
  int_t *perm = NULL,*degree = NULL;
  vertexlist **C = NULL, **Cdeg = NULL, *V = NULL, *E = NULL, *Cu, *Cv;
  vertex *w, *z, *u, *v, *v0, *Vl = NULL;
  adjgraph *Ac = NULL;

  n = A->numVertices;
  if (startvertex < 0 || startvertex > n-1) goto cleanup;

  // find max vertex degree
  for (i=0;i<n;i++) 
    degree_max = MAX(degree_max, A->adjlist[i]->size);
#ifdef DEBUG
  printf("debug: degree_max = %li\n", degree_max);
#endif
  
  // allocate and initialize V
  V = vertexlist_new();
  if (!V) goto cleanup;
  Vl = calloc(n,sizeof(vertex));
  if (!Vl) goto cleanup;
  V->head = Vl;
  V->tail = V->head + n - 1;
  V->size = n;
  v = V->head;
  v->prev = NULL;
  for (i=0;i<n-1;i++) {
    v->next = v + 1;
    v->value = i;
    v->next->prev = v;
    v = v->next;
  }
  v->value = n - 1;
  v->next = NULL;

  // allocate and initialize C and Cdeg
  C = calloc(n,sizeof(vertexlist *));
  Cdeg = calloc(degree_max+1,sizeof(vertexlist *));
  if (!C || !Cdeg) goto cleanup;

  for (i=0;i<n;i++) {if (!(C[i] = vertexlist_new())) goto cleanup;}
  Cdeg[0] = V;  
  for (i=1;i<=degree_max;i++) {if (!(Cdeg[i] = vertexlist_new())) goto cleanup;}

  // allocate perm and degree
  perm = malloc(n*sizeof(int_t));
  if (!perm) goto cleanup;

  degree = calloc(n,sizeof(int_t));
  if (!degree) goto cleanup;

  // MAXCHORD initialize
  v0 = Cdeg[0]->head + startvertex;
  perm[n-1] = startvertex;
  vertexlist_pop(Cdeg[0], v0);
  degree[startvertex] = -1; // indicates that startvertex has been eliminated

  // Main loop
  for (k = n-1; k>0; k--) {
    // step 2: for u in V_k with (u,v0) in E
    //             if C(u) in C(v0) then 
    //                  C(u) := C(u) + {v0}

    E = A->adjlist[v0->value];
    u = E->head;

    for (i=0;i<E->size;i++) {

      // if u in V_k with (u,v0) in E
      if (degree[u->value] >= 0) {
	// check if C(u) in C(v0)
	Cu = C[u->value]; Cv = C[v0->value];
	if (Cv->size < Cu->size) {
	  u = u->next;
	  continue;
	}
	
#ifdef DEBUG
	printf("debug: C(%i): ",u->value);
	w = Cu->head;
	for (j=0;j<Cu->size;j++) {
	  printf(" %i", w->value);
	  w = w->next;
	} 
	printf("\ndebug: C(%i): ",v0->value);
	w = Cv->head;
	for (j=0;j<Cv->size;j++) {
	  printf(" %i", w->value);
	  w = w->next;
	}
	printf("\n");
#endif

	w = Cu->head; z = Cv->head;
	if (w != NULL) {
	  for (j=0;j<Cu->size;j++) {
	    while (z != NULL) {
	      if (w->value == z->value) break;
	      z = z->next;
	    }
	    if (z == NULL) break;
	    //z = z->next;
	    w = w->next;
	  }
	  if (z == NULL) {
	    u = u->next;
	    continue;
	  }
	}

	// insert v0 in C(u)
#ifdef DEBUG
	printf("debug: inserting %i in C(%i)\n", v0->value, u->value);
#endif	
	if (vertexlist_insert(Cu,v0->value) == -1) goto cleanup;

	// update Cdeg and degree
	z = Cdeg[degree[u->value]]->head;
	for (j=0;j<Cdeg[degree[u->value]]->size;j++) {	  
	  if (z->value == u->value) break;
	  z = z->next;
	}

	vertexlist_pop(Cdeg[degree[u->value]], z);
	degree[u->value] += 1;
	vertexlist_push(Cdeg[degree[u->value]], z);

      }
      u = u->next;
    }

    // step 3: select v0 in V_k such that |C(v0)| >= |C(v)| for all v in V_k
    for (j=0;j<=degree_max;j++) {
      if (Cdeg[degree_max - j]->size > 0) {
	//v0 = Cdeg[degree_max - j]->head;
	v0 = Cdeg[degree_max - j]->tail;
	break;
      }
    } 

    // step 4: set perm(k-1) = v0, V_{k-1} = V_k \ v0
    perm[k-1] = v0->value;
    vertexlist_pop(Cdeg[degree_max - j], v0);
    degree[v0->value] = -1;
  }

  // reorganize nodes (move upper triangular entries to lower triangle)
  for (i=0;i<n;i++) {
    u = C[i]->head;
    while (u!=NULL) {
      if (u->value < i) {
	v = u->next;
	vertexlist_pop(C[i],u);
	l = u->value;
	u->value = i;
	vertexlist_push(C[l],u);
	u = v;		       
      }
      else {
	u = u->next;
      }
    }
  }

  // create adjgraph
  Ac = malloc(sizeof(adjgraph)); 
  if (!Ac) goto cleanup;
  Ac->adjlist = C;
  Ac->numVertices = n;
  compl = 1;

  // Clean up
 cleanup:
  if (V) free(V);
  if (Vl) free(Vl);
  if (Cdeg) {
    for (i=1;i<=degree_max;i++)
      if (Cdeg[i]) free(Cdeg[i]);
    free(Cdeg); 
  }
  if (degree) free(degree);  
  if (compl) {
    *order = perm;
    return Ac;
  }
  else {
    *order = NULL;
    if (Ac) adjgraph_destroy(Ac);    
    if (perm) free(perm);
    return NULL; 
  }
}

