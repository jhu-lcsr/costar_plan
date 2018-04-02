import argparse
import os
import numpy as np
import h5py 
import networkx as nx
import matplotlib.pyplot as plt

'''
Usage
python plot_graph --path <path to data folder> --name <name of action > --start <int> --end <int> --ignore_failure <bool>
python plot_graph --path 'data/' 
default choose all files and ignore failure case and action name is "labels_to_name"
'''


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--name",type=str, default=unicode("labels_to_name"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=100000)
    parser.add_argument("--ignore_failure", type = bool, default=True)
    return vars(parser.parse_args())

def main(args,root="root"):
	# init graph
	graph = nx.DiGraph()
	node_list = set()
	node_list.add(root)

	# Read data
	for filename in os.listdir(args['path']): 
		if filename.startswith('.'):
			continue
		idx = int(filename[7:13])
		if idx < args['start'] or idx > args['end']:
			continue
		
		if args['ignore_failure']:
			flag = filename[14:-4]
			if flag is 'failure':
				continue

		data = h5py.File(args['path']+filename,'r')
		labels = list(data[args['name']])
		prev_label = root

		for label in labels:
			if not label == prev_label:
				graph.add_edge(prev_label,label,weight=1)
			prev_label = label
			node_list.add(label)

	pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
	nx.draw_networkx_edges(graph, pos, width=1.0, alpha=1., arrows=False)
	nx.draw(graph, pos, prog='dot', node_size=1000, nodelist=node_list,
            width=1.0, alpha=1., arrows=True, with_labels=True,)

	plt.axis
	plt.show()

if __name__ == "__main__":
	args = _parse_args()
	main(args)