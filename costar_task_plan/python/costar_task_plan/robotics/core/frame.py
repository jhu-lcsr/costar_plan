
import os


class Frame(object):

    def __init__(self, name, obj_class, tf_frame, namespace=""):
        self.name = name
        self.obj_class = obj_class
        self.tf_frame = tf_frame

        self.tf_name = os.path.join(namespace, name)
