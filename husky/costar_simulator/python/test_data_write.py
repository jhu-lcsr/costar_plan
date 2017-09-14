import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Pose, Twist
import numpy as np
from npz import NpzDataset



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
    img_np = imageToNumpy(data)
    
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


if __name__ == '__main__':
    listener()
    global gazeboModels
    global robot_action
    global robot_pose
    global img_np
    goalPub = rospy.Publisher(huskyGoalTopic, PoseStamped, queue_size=1)
    seqNumber = 0
    npz_writer = NpzDataset('/home/katyakd1/projects/costar_ws/src/costar_plan/husky/costar_simulator/python/data/')
    
    while not rospy.is_shutdown():
        
        rate = rospy.Rate(10) # 10hz
        current_example = {}
        current_example['reward'] = list()
        current_example['done'] = list()
        current_example['example'] = list()
        current_example['label'] = list()
        current_example['multi_feature'] = list()        
        
        
        for objectName in poseDictionary:
            poseStampedMsg = PoseStamped()
            poseStampedMsg.header.frame_id = "map"
            poseStampedMsg.header.stamp = rospy.Time.now()
            poseStampedMsg.pose = poseDictionary[objectName]
            
            #print "position is ", poseStampedMsg.pose           
            goalPub.publish(poseStampedMsg)
            
            iterations = 0
            while (goalReached(poseStampedMsg.pose) == False and iterations < 150):
                #print "sleeping ", iterations
                rate.sleep()
                # get global variables and write
                current_example['reward'].append(0)
                if iterations == 149:
                    current_example['done'].append(1)
                else:
                    current_example['done'].append(0)
                current_example['example'].append(seqNumber)
                current_example['label'].append(list(poseDictionary.keys()).index(objectName))
                
                # now add the features
                current_example['multi_feature'].append([img_np, robot_pose, robot_action])
                iterations = iterations + 1
            
            #print ("writing sample")
            npz_writer.write(current_example, seqNumber, 1) 
            seqNumber = seqNumber + 1
        
            
    
    
    #rospy.spin()
    