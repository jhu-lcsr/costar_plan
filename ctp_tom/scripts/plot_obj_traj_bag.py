#!/usr/bin/env python

'''
Plot the trajectory of all the objects and hands
rosrun ctp_tom plot_obj_traj_bag --filename <rosbag file path> --ignore_inputs
'''

import argparse
import numpy as np
import rosbag
import rospy


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename","-f",
                        type=str,
                        default="data.bag")
    parser.add_argument("--ignore_inputs",
                        help="Do not plot controllers and head.",
                        action="store_true")
    return vars(parser.parse_args())

'''
Modify to dealing with different cases
'''
def getHandName(hand,name):
    #if not hand.object_in_hand is None:
    name += '_with_' + hand.object_in_hand
    #if not hand.object_acted_on is None:
    name += '_to_' + hand.object_acted_on


def _main(filename, ignore_inputs, **kwargs):
 	obj_history = {}
 	left_hand_history = []
 	right_hand_history = []
 	demo_topic = "/vr/learning/getDemonstrationInfo"
 	alias_topic = "/vr/learning/getActivityAlias"
 	bag = rosbag.Bag(filename)

 	obj_ignore = {'surveillance_camera','Head (eye)','Controller (left)','Controller (right)','high_table'}
 	for topic, msg, _ in bag:
 		if topic == alias_topic:
 			print("Alias: ",msg.old_name,msg.new_name)
 			continue
 		if topic == demo_topic:
 			# get objects
 			for obj in msg.object:
 				name = obj.name
				if ignore_inputs:
					if 'Hand' in name or 'Controller' in name:
 						continue
 				if not name in obj_history:
 					obj_history[name] = []
 				obj_history[name].append([obj.pose.position.x, obj.pose.position.y, obj.pose.position.z])

 			# get hands
 			hand = msg.left
 			name = 'left_hand'
 			getHandName(hand,name)
 			if not name in obj_history:
 				obj_history[name] = []
 			obj_history[name].append([hand.pose.position.x, hand.pose.position.y, hand.pose.position.z])

			hand = msg.right
 			name = 'right_hand'
 			getHandName(hand,name)
 			if not name in obj_history:
 				obj_history[name] = []
 			obj_history[name].append([hand.pose.position.x, hand.pose.position.y, hand.pose.position.z])

 	
 	
 	# plot
 	fig = plt.figure()
 	ax = fig.gca(projection = '3d')
 	for obj, data in obj_history.items():
 		data = np.array(data)
 		ax.plot(data[:,0], data[:,1], data[:,2], label=obj)
 	plt.title('Object Positons')
 	ax.legend()
 	plt.show()

if __name__ == "__main__":
	args = _parse_args()
	_main(**args)





