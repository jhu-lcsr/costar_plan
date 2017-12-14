from __future__ import print_function

import networkx as nx
import numpy as np

def showTask(task, root="ROOT()", filename="task.dot"):
    import matplotlib.pyplot as plt

    g = nx.DiGraph()

    nodes = [root]
    visited = set()
    nodelist = []

    while len(nodes) > 0:
        node = nodes.pop()
        visited.add(node)
        children = task.children[node]
        weights = task.weights[node]
        for child, wt in zip(children, weights):
            g.add_edge(node, child, weight=wt)
            nodelist.append(child)
            if child not in visited:
                nodes.append(child)

    pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    nx.draw_networkx_edges(g, pos, width=1.0, alpha=1., arrows=False)
    nx.draw(g, pos, prog='dot', node_size=1000, nodelist=nodelist,
            width=1.0, alpha=1., arrows=True, with_labels=True,)
    labels = nx.get_edge_attributes(g,'weight')
    nx.draw_networkx_edge_labels(g,pos,edge_labels=labels)
    #a = nx.nx_agraph.to_agraph(g)
    #a.draw('ex.png', prog='dot')
    plt.axis('off')
    plt.show()

