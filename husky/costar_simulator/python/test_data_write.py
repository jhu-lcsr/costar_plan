import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Pose, Twist
import numpy as np
from npz import NpzDataset
import tf
from tf import TransformListener

from scipy.misc import imresize
import cv2
from cv_bridge import CvBridge, CvBridgeError




gazeboModelsTopic = "/gazebo/model_states"
overheadImageTopic = "/overhead/camera/image_raw"
huskyGoalTopic = "/move_base_simple/goal"
huskyOdomTopic = "/husky_velocity_controller/odom"
huskyCmdVelTopic = "/husky_velocity_controller/cmd_vel"

dumpsterPose = Pose()
dumpsterPose.position.x = -1.135
dumpsterPose.position.y = -3.856
dumpsterPose.position.z = 0
dumpsterPose.orientation.x = 0
dumpsterPose.orientation.y = 0
dumpsterPose.orientation.z = .959
dumpsterPose.orientation.w = -0.2847

barrierPose = Pose()
barrierPose.position.x = 1.325
barrierPose.position.y = -0.89
barrierPose.position.z = 0
barrierPose.orientation.x = 0
barrierPose.orientation.y = 0
barrierPose.orientation.z = .459
barrierPose.orientation.w = 0.888

constructionBarrelPose = Pose()
constructionBarrelPose.position.x = 1.0088
constructionBarrelPose.position.y = -4.96
constructionBarrelPose.position.z = 0
constructionBarrelPose.orientation.x = 0
constructionBarrelPose.orientation.y = 0
constructionBarrelPose.orientation.z = -0.5729
constructionBarrelPose.orientation.w = 0.81957

fireHydrantPose = Pose()
fireHydrantPose.position.x = 4.02
fireHydrantPose.position.y = -5.114
fireHydrantPose.position.z = 0
fireHydrantPose.orientation.x = 0
fireHydrantPose.orientation.y = 0
fireHydrantPose.orientation.z = 0.8945
fireHydrantPose.orientation.w = -0.447


#store the positions as ints, have a dictionary with keys, robots pose, orientation, and action, sequence (trial number), top down image is 5
#randomize the ordering of targets
poseDictionary = {'Dumpster': dumpsterPose, 'Barrier' : barrierPose, 'Barrel' :  constructionBarrelPose, 'Fire Hydrant': fireHydrantPose}

dumpsterPose = None
barrierPose = None

#globals
gazeboModels = None
img_np = None
robot_pose = Odometry()
robot_action = Twist()

name_to_dtypes = {
	"rgb8":    (np.uint8,  3),
	"rgba8":   (np.uint8,  4),
	"rgb16":   (np.uint16, 3),
	"rgba16":  (np.uint16, 4),
	"bgr8":    (np.uint8,  3),
	"bgra8":   (np.uint8,  4),
	"bgr16":   (np.uint16, 3),
	"bgra16":  (np.uint16, 4),
	"mono8":   (np.uint8,  1),
	"mono16":  (np.uint16, 1),}

def imageToNumpy(msg):
	if not msg.encoding in name_to_dtypes:
		raise TypeError('Unrecognized encoding {}'.format(msg.encoding))
	
	dtype_class, channels = name_to_dtypes[msg.encoding]
	dtype = np.dtype(dtype_class)
	dtype = dtype.newbyteorder('>' if msg.is_bigendian else '<')
	shape = (msg.height, msg.width, channels)

	data = np.fromstring(msg.data, dtype=dtype).reshape(shape)
	data.strides = (
		msg.step,
		dtype.itemsize * channels,
		dtype.itemsize
	)

	if channels == 1:
		data = data[...,0]
	return data


def imageCallback(data):
    global img_np
    img_np_bk = imageToNumpy(data)
    #print img_np.shape
    img_np = imresize(img_np_bk, (64, 64))
   
    #print img_np.shape
    
def odomCallback(data):
    global robot_pose
    robot_pose = data
    
    
def cmdVelCallback(data):
    global robot_action
    robot_action = data

def modelsCallback(data):
    global gazeboModels
    gazeboModels = data    
    

def listener():
    # setup subscribers
    rospy.init_node('costar_simulator_husky', anonymous=True)
    rospy.Subscriber(overheadImageTopic, Image, imageCallback)
    rospy.Subscriber(huskyOdomTopic, Odometry, odomCallback)
    rospy.Subscriber(huskyCmdVelTopic, Twist, cmdVelCallback)


    
# returns true if goal has been reached, else false    
def goalReached(finalPose):
    #TODO
    return False
      
def getObjectList():
    return list(poseDictionary.keys())

def getOverheadCamera():
    pass

def moveToObject(objectName):
    pass

def finishCurrentExample(example, max_label=-1, seed=None):
        '''
        Preprocess this particular example:
        - split it up into different time windows of various sizes
        - compute task result
        - compute transition points
        - compute option-level (mid-level) labels
        '''
        #print("Finishing example",example,seed)

        # ============================================================
        # Split into chunks and preprocess the data.
        # This may require setting up window_length, etc.
        next_list = ["reward", "done", "example", "label", "image", "robot_pose", "robot_action"]
        # -- NOTE: you can add other features here in the future, but for now
        # we do not need these. Label gets some unique handling.
        prev_list  = []
        first_list = ["image", "robot_pose"]
        goal_list = ["image", "robot_pose"]
        length = len(current_example['label'])

        # Create an empty dict to hold all the data from this last trial.
        data = {}
        data["prev_label"] = []

        # Compute the points where the data changes from one label to the next
        # and save these points as "goals".
        switches = []
        count = 1
        label = current_example['label']
        for i in xrange(length):
            if i+1 == length:
                switches += [i] * count
                count = 1
            elif not label[i+1] == label[i]:
                switches += [i+1] * count
                count = 1
            else:
                count += 1
        print (len(switches),  len(current_example['example']))
        assert(len(switches) == len(current_example['example']))

        # ============================================
        # Set up total reward
        total_reward = np.sum(current_example["reward"])
        data["value"] = [total_reward] * len(current_example["example"])

        # ============================================
        # Loop over all entries. For important items, take the previous frame
        # and the next frame -- and possibly even the final frame.
        prev_label = max_label
        for i in xrange(length):
            i0 = max(i-1,0)
            i1 = min(i+1,length-1)
            ifirst = 0

            # We will always include frames where the label changed. We may or
            # may not include frames where the 
            if current_example["label"][i0] == current_example["label"][i1] \
                    and not i0 == 0 \
                    and not i1 == length - 1 \
                    and not np.random.randint(2) == 0:
                        continue

            # ==========================================
            # Finally, add the example to the dataset
            for key, values in current_example.items():
                if not key in data:
                    data[key] = []
                    if key in next_list:
                        data["next_%s"%key] = []
                    if key in prev_list:
                        data["prev_%s"%key] = []
                    if key in first_list:
                        data["first_%s"%key] = []
                    if key in goal_list:
                        data["goal_%s"%key] = []

                # Check data consistency
                if len(data[key]) > 0:
                    if isinstance(values[0], np.ndarray):
                        assert values[0].shape == data[key][0].shape
                    if not type(data[key][0]) == type(values[0]):
                        print(key, type(data[key][0]), type(values[0]))
                        raise RuntimeError('Types do not match when' + \
                                           ' constructing data set.')

                # Append list of features to the whole dataset
                data[key].append(values[i])
                #if key == "label":
                #    data["prev_%s"%key].append(prev_label)
                #    prev_label = values[i]
                if key in prev_list:
                    data["prev_%s"%key].append(values[i0])
                if key in next_list:
                    data["next_%s"%key].append(values[i1])
                if key in first_list:
                    data["first_%s"%key].append(values[ifirst])
                if key in goal_list:
                    data["goal_%s"%key].append(values[switches[i]])

        # ===================================================================
        # Print out the seed associated with this example for reproduction, and
        # use it as part of the filename. If the seed is not provided, we will
        # set to the current example index.
        if seed is None:
            seed = example
        #print ("writing ", current_example['example'][0])
            
        #print ("data prev_label ", data['prev_label'], " label ", data['label'])
        #print ("goal robot pose ", data['goal_robot_pose'])
        #from scipy.misc import imsave    
        #imsave('/tmp/rgb_gradient.png', data['goal_image'][0])
        

        npz_writer.write(data, current_example['example'][0], total_reward)
    
        
if __name__ == '__main__':
    listener()
    global gazeboModels
    global robot_action
    global robot_pose
    global img_np
    goalPub = rospy.Publisher(huskyGoalTopic, PoseStamped, queue_size=1)
    seqNumber = 0
    npz_writer = NpzDataset('/home/katyakd1/projects/costar_ws/src/costar_plan/husky/costar_simulator/python/data/')
    prev_label = -1
    while not rospy.is_shutdown():
        
        rate = rospy.Rate(10) # 10hz
        
        
        for objectName in poseDictionary:
            poseStampedMsg = PoseStamped()
            poseStampedMsg.header.frame_id = "map"
            poseStampedMsg.header.stamp = rospy.Time.now()
            poseStampedMsg.pose = poseDictionary[objectName]
            
            #print "position is ", poseStampedMsg.pose           
            goalPub.publish(poseStampedMsg)
            
            current_example = {}
            current_example['reward'] = list()
            current_example['done'] = list()
            current_example['example'] = list()
            current_example['label'] = list()
            current_example['prev_label'] = list()
            current_example['image'] = list()  
            current_example['robot_pose'] = list()
            current_example['robot_action'] = list()
        
        
            
            iterations = 0
            
            while (goalReached(poseStampedMsg.pose) == False and iterations < 150):
                #print "sleeping ", iterations
                rate.sleep()
                # get global variables and write
                current_example['reward'].append(1)
                if iterations == 149:
                    current_example['done'].append(1)
                else:
                    current_example['done'].append(0)
                current_example['example'].append(seqNumber)
                
                current_example['image'].append(img_np)
                quaternion = (robot_pose.pose.pose.orientation.x, robot_pose.pose.pose.orientation.y,robot_pose.pose.pose.orientation.z, robot_pose.pose.pose.orientation.w)
                euler = tf.transformations.euler_from_quaternion(quaternion)
                roll = euler[0]
                pitch = euler[1]
                yaw = euler[2]
                current_example['robot_pose'].append([robot_pose.pose.pose.position.x, \
                robot_pose.pose.pose.position.y, robot_pose.pose.pose.position.z, roll, pitch, yaw])
                
                
                current_example['robot_action'].append([robot_action.linear.x, robot_action.linear.y, \
                robot_action.linear.z, robot_action.angular.x, robot_action.angular.y, robot_action.angular.z])
                
                current_example['label'].append(poseDictionary.keys().index(objectName))
                current_example['prev_label'].append(prev_label)
                
                
                
                iterations = iterations + 1
            
            #print ("writing sample")
            finishCurrentExample(current_example)
            #npz_writer.write(current_example, seqNumber, 1) 
            seqNumber = seqNumber + 1
            print "prev_label", prev_label
            prev_label = poseDictionary.keys().index(objectName)
            print "prev_label after update", prev_label

        
            
    
    
    #rospy.spin()
    