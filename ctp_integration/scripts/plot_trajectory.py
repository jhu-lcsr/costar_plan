import argparse
import numpy as np
import h5py 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

'''
Usage
python plot_trajectory --filename <path to file>

Due to my configuration, I need to use python2 instead of python
I think life will be easier in Linux enviroment
'''


def _parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--filename","-f",
                        type=str)
	return vars(parser.parse_args())

def main(args):
	data = h5py.File(args['filename'],'r')
	label = np.array(data[u'label'])
	arm = np.array(data[u'arm'])

	time_length = len(arm)
	action_trajectories = {}
	for i in range(time_length):
		action = label[i]
		if not action in action_trajectories:
			action_trajectories[action] = []
		action_trajectories[action].append(arm[i][0:3])



	fig = plt.figure()
 	ax = fig.gca(projection = '3d')
 	for action, trajectory in action_trajectories.items():
 		trajectory = np.array(trajectory)
 		ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], label=str(action))
 	ax.legend()
 	plt.show()

if __name__ == "__main__":
	args = _parse_args()
	main(args)