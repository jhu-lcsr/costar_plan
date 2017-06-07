
import rospy


class DetectedObject(object):

    def __init__(self, name, obj_class, obj, t=rospy.Time(0)):
        self.name = name
        self.obj_class = obj_class
        self.obj = obj
        self.t = t
