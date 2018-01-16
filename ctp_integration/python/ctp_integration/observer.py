
import rospy


class Observer(object):

    def __init__(self, world, detect_srv, topic):
        self.world = world
        self.detect_srv = detect_srv
        self.detected_objects_topic = topic

    def _cb(self, msg):
        oass

    def __call__(self):
        # Call the detect objects service and wait for response
        world = self.world.fork()
        self.detect_srv()
        
