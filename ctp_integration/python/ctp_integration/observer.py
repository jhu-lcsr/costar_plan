
import rospy
import tf

class IdentityObserver(object):
    def __init__(self, world, task):
        self.world = world
        self.task = task

    def __call__(self):
        return self.world, self.task

class Observer(object):

    def __init__(self, world, task, detect_srv, topic, tf_listener):
        self.world = world
        self.task = task
        self.detect_srv = detect_srv
        self.detected_objects_topic = topic
        self.msg = None
        if tf_listener is not None:
            self.tf_listener = tf_listener
        else:
            self.tf_listener = tf.TransformListener()

    def _cb(self, msg):
        self.msg = msg

    def __call__(self):
        # Call the detect objects service and wait for response
        world = self.world.fork()
        self.detect_srv()
        
