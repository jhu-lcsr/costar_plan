import PyKDL as kdl
import rospy
import tf

class TransformIntegator(object):

    def __init__(self, name, root, transforms={}, listener=None):
        self.name = name
        self.transforms = transforms
        self.root = root

        if listener is not None:
            self.listener = listener
        else:
            self.listener = tf.TransformListener()

    def addTransform(self, name, pose):
        self.transforms[name] = pose

    def tick(self):
        '''
        Look for all transforms in the list
        '''

        if not self.listener.frameExists(self.root):
            raise RuntimeError('failed to find root frame')

        for name, pose in self.transforms:
            if self.listener.frameExists(name):
                t = self.listener.getLatestCommonTime(self.root, name)
                print name, t, root
                p, q = self.listener.lookupTransform(self.root, name, t)


