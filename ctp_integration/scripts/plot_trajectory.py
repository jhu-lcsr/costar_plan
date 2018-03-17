import argparse
import numpy as np
import h5py 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

'''
Usage
python plot_trajectory --filename <path to file> 

If data collector is modified, use extra arguments to modify
pose_name:  topic name of pose in h5f data
action_name:  topic of action name
label_name: topic of label 


Due to my configuration, I need to use python2 instead of python
I think life will be easier in Linux enviroment
'''


def _parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--filename","-f", type=str)
	parser.add_argument("--pose_name",type=str,default = unicode("pose"))
	parser.add_argument("--action_name",type=str,default = unicode("labels_to_name"))
	parser.add_argument("--label_name",type=str,default = unicode("label"))
	return vars(parser.parse_args())

def main(args):
	data = h5py.File(args['filename'],'r')
	label = np.array(data[args['label_name']])
	action_name = np.array(data[args['action_name']])
	pose = np.array(data[args['pose_name']])
	time_length = len(pose)
	action_trajectories = {}
	action_idx = -1
	pre_label = -1
	for i in range(time_length):
		
		if not label[i] == pre_label:
			action_idx = action_idx + 1
		action = action_name[action_idx]
		pre_label = label[i]

		if not action in action_trajectories:
			action_trajectories[action] = []
		action_trajectories[action].append(pose[i][0:3])



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