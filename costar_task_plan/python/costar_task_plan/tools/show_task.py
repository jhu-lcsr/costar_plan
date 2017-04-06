
import matplotlib.pyplot as plt
import networkx as nx

def showTask(task,root="ROOT()",filename="task.dot"):

  g = nx.DiGraph()

  nodes = [root]

  while len(nodes) > 0:
    node = nodes.pop()
    children = task.children[node]
    for child in children:
      g.add_edge(node, child)
    nodes += children

  nx.write_dot(g, filename)
  pos=nx.graphviz_layout(g,prog='dot')
  nx.draw(g, pos, node_size=1000, width=1.0, alpha=1., arrows=False)

  plt.axis('off')
  plt.show()

