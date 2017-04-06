
import networkx as nx

def makeGraph(root):

  g = nx.DiGraph()
  nid = 0
  name, nid = get_name(root, nid)
  g.add_node(name)

  good = []
  bad = []
  mid = [name]

  for child in root.children:
    nid = add_to_graph(g, name, child, good, bad, mid, nid)

  return g, good, bad, mid

def get_name(node, nid):
  if node.state is None:
    t = 0
  else:
    t = int(node.state.t)
  name = "%s %d"%(node.tag, nid)
  return name, nid+1

def add_to_graph(g, parent_name, node, good, bad, mid, nid):
  name, nid = get_name(node, nid)
  g.add_edge(parent_name, name)

  # loop over conditions and determine if this is in a list
  if not node.terminal:
    mid.append(name)
  else:
    world = node.world
    # loop over conditions
    for condition, wt, cname in world.conditions:
      actor = world.actors[0]
      state = actor.state
      prev_state = actor.last_state

      if not condition(world, state, actor, prev_state):
        if wt < 0:
          bad.append(name)
        else:
          good.append(name)

  for child in node.children:
    nid = add_to_graph(g, name, child, good, bad, mid, nid)

  return nid

