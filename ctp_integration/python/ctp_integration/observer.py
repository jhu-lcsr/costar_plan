
import rospy
import tf2_ros as tf2

from costar_objrec_msgs.msg import DetectedObjectList

class IdentityObserver(object):
    def __init__(self, world, task):
        self.world = world
        self.task = task

    def __call__(self):
        return self.world, self.task

class Observer(object):
    """ Runs the object detection algorithm and assembles the list of detected objects.
    """

    def __init__(self, world, task, detect_srv, topic,
                 tf_buffer=None,
                 tf_listener=None,
                 verbose=0):
        '''
        Create an observer. This will take a world and other information and
        use it to provide updated worlds.
        '''
        self.world = world
        self.task = task
        self.detect_srv = detect_srv
        self.detected_objects_topic = topic
        self.msg = None
        if tf_buffer is None:
            self.tf_buffer = tf2.Buffer()
        else:
            self.tf_buffer = tf_buffer
        if tf_listener is None:
            self.tf_listener = tf2.TransformListener(self.tf_buffer)
        else:
            self.tf_listener = tf_listener

        self._detected_objects_sub = rospy.Subscriber(
                self.detected_objects_topic, 
                DetectedObjectList,
                self._detected_objects_cb)
        self.verbose = verbose

    def _detected_objects_cb(self, msg):
        # Save detected objects message
        self.msg = msg

    def __call__(self):

        # Empty out the current version of the task to get a new task model
        # TODO(ahundt) figure out why this was here.
        # self.task.clear()

        # Call the detect objects service and wait for response
        #world = self.world.fork()
        self.detect_srv()

        # Spin
        rate = rospy.Rate(10)
        while self.msg == None and not rospy.is_shutdown():
            rate.sleep()

        # Generate a task plan from a message
        # Step 1. Create args describing which objects we saw.
        args = {}
        for obj in self.msg.objects:
            if self.verbose:
                rospy.loginfo("Observer Objects: " + str(self.msg.objects))
            name = obj.id
            obj_class = obj.object_class
    
            # Create arguments for the task plan
            if not obj_class in args:
                args[obj_class] = set()
            args[obj_class].add(name)
        
        if self.verbose:
            rospy.loginfo("Detected objects: " + str(args))

        # Step 2. Compile the plan.
        #self.world.addObjects(args)
        #filled_args = self.task.compile(args)
        #print(filled_args)

        # Env is the wrapper that interfaces with the world and consumes
        # our commands
        # env = None # TODO: add this
        return self.task, self.world
        
