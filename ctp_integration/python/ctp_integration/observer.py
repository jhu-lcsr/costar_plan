
import rospy
import tf

from costar_objrec_msgs.msg import DetectedObjectList

class IdentityObserver(object):
    def __init__(self, world, task):
        self.world = world
        self.task = task

    def __call__(self):
        return self.world, self.task

class Observer(object):

    def __init__(self, world, task, detect_srv, topic, tf_listener=None):
        '''
        Create an observer. This will take a world and other information and
        use it to provide updated worlds.
        '''
        self.world = world
        self.task = task
        self.detect_srv = detect_srv
        self.detected_objects_topic = topic
        self.msg = None
        if tf_listener is not None:
            self.tf_listener = tf_listener
        else:
            self.tf_listener = tf.TransformListener()

        self._detected_objects_sub = rospy.Subscriber(
                self.detected_objects_topic, 
                DetectedObjectList,
                self._detected_objects_cb)

    def _detected_objects_cb(self, msg):
        self.msg = msg

    def __call__(self):
        # Call the detect objects service and wait for response
        world = self.world.fork()
        self.detect_srv()

        rate = rospy.Rate(10)
        while self.msg == None and not rospy.is_shutdown():
            rate.sleep()

        # Yeah just wait for a moment until this is done
        rospy.sleep(0.5)
    
        # Env is the wrapper that interfaces with the world and consumes
        # our commands
        env = None # TODO: add this
        return self.task, world
        
