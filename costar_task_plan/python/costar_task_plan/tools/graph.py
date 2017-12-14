
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
    name = "%s %d" % (node.tag, nid)
    return name, nid + 1


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


def showGraph(root, filename='test.dot'):
    import matplotlib.pyplot as plt
    g, good, bad, mid = makeGraph(root)
    plt.figure(figsize=(10, 10), dpi=80)
    nx.write_dot(g, filename)

    # same layout using matplotlib with no labels
    pos = nx.graphviz_layout(g, prog='dot')

    nx.draw_networkx_edges(g, pos, width=1.0, alpha=1., arrows=False)
    ALPHA = 1.0
    colors = [(0.2, 0.8, 0.2)] * len(good)
    nx.draw_networkx_nodes(g, pos,
                           nodelist=good,
                           node_color=colors,
                           alpha=ALPHA,

                           node_shape='s',
                           node_size=1600)
    colors = [(0.9, 0.4, 0.4)] * len(bad)
    nx.draw_networkx_nodes(g, pos,
                           nodelist=bad,
                           node_color=colors,
                           alpha=ALPHA,
                           node_shape='8',
                           node_size=1600)
    colors = [(0.8, 0.8, 0.8)] * len(mid)
    nx.draw_networkx_nodes(g, pos,
                           nodelist=mid,
                           node_color=colors,
                           node_shape='s',
                           alpha=ALPHA,
                           node_size=1600)
    labels = {}
    lookup = {
        "NODE": "0",
        "Default": "D",
        "Left": "L",
        "Right": "R",
        "Pass": "P",
        "Stop": "S",
        "Wait": "W",
        "Follow": "F",
        "Finish": "C",
    }
    for name in good:
        labels[name] = lookup[name.split(' ')[0]]
    for name in bad:
        labels[name] = lookup[name.split(' ')[0]]
    for name in mid:
        labels[name] = lookup[name.split(' ')[0]]
    nx.draw_networkx_labels(g, pos, labels, font_size=20)

    plt.axis('off')
    plt.show()
