
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


def showTask(task, root="ROOT()", filename="task.dot"):

    g = nx.DiGraph()

    nodes = [root]
    visited = set()

    while len(nodes) > 0:
        node = nodes.pop()
        visited.add(node)
        children = task.children[node]
        for child in children:
            g.add_edge(node, child)
            if child not in visited:
                nodes.append(child)

    pos = graphviz_layout(g)
    nx.draw(g, pos, prog='dot', node_size=1000,
            width=1.0, alpha=1., arrows=False)

    plt.axis('off')
    plt.show()
