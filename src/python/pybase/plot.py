try:
    import networkx as nx
    __nx__ = True
except:
    __nx__ = False

try:
    import numpy as np
    __np__ = True
except:
    __np__ = False

try:
    import matplotlib as mpl
    __mpl__ = True
except:
    __mpl__ = False
    
try:
    from scipy import sparse as scipy_sparse
    __scipy__ = True
except:    
    __scipy__ = False

if __mpl__ and __np__ and __scipy__:
    def spy(symb, reordered = True, symmetric = True, **kwargs):
        """
        Draw sparsity pattern.

        Requires: Matplotlib, Scipy, and Numpy
        """
        A = symb.sparsity_pattern(reordered=reordered,symmetric=symmetric)
        colptr,rowidx,val = A.CCS
        M = scipy_sparse.csc_matrix((np.array(list(val)),np.array(list(rowidx)),np.array(list(colptr))))
        fig = mpl.pyplot.spy(M, **kwargs)
        return fig
    
if __nx__ and __mpl__:
    def sparsity_graph(symb, **kwargs):
        """
        Draw sparsity graph.

        Requires: NetworkX and Graphviz
        """
        layout = kwargs.get('layout',None)
        G = nx.Graph()
        G.add_nodes_from(range(symb.n))
        for k in range(symb.Nsn):
            ck = symb.snrowidx[symb.sncolptr[k]:symb.sncolptr[k+1]]
            G.add_edges_from([(ck[i],ck[j]) for i in range(len(ck)) for j in range(i,len(ck))])
        if layout is None:
            pos=nx.graphviz_layout(G)
        else:
            pos = layout(G)
        fig = nx.draw(G, pos,**kwargs)
        return fig

if __nx__ and __mpl__:
    def etree_graph(symb, **kwargs):
        """
        Draw supernodal elimination tree.

        Requires: NetworkX and Graphviz
        """
        layout = kwargs.get('layout',None)
        G = nx.DiGraph()
        G.add_nodes_from(range(symb.Nsn))
        G.add_edges_from([(symb.snpar[k],k) for k in range(symb.Nsn) if symb.snpar[k] != k ])
        if layout is None:
            pos=nx.graphviz_layout(G, prog='dot')
        else:
            pos = layout(G)
        fig = nx.draw(G,pos,**kwargs)
        return fig
