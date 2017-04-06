
import matplotlib.pyplot as plt
import networkx as nx

def showTask(task,root="ROOT()",filename="task.dot"):

  g = nx.DiGraph()

  nodes = [root]

  while len(nodes) > 0:
    node = nodes.pop()
    children = task.children[node]
    print node, children
    for child in children:
      g.add_edge(node, child)
    nodes += children

  nx.write_dot(g, filename)
  pos=nx.graphviz_layout(g,prog='dot')
  nx.draw(g, pos)

  plt.axis('off')
  plt.show()

