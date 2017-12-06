import numpy as np
import PyKDL as kdl
import rospy
import tf
import tf_conversions.posemath as pm

class TransformIntegator(object):

    def __init__(self, name, root, transforms={}, listener=None, broadcaster=None):
        self.name = name
        self.transforms = transforms
        self.root = root

        if listener is not None:
            self.listener = listener
        else:
            self.listener = tf.TransformListener()
        if broadcaster is not None:
            self.broadcaster = broadcaster
        else:
            self.broadcaster = tf.TransformBroadcaster()

    def addTransform(self, name, pose):
        self.transforms[name] = pose

    def tick(self):
        '''
        Look for all transforms in the list
        '''

        if not self.listener.frameExists(self.root):
            return

        count = 0
        avg_p = np.zeros(3)
        avg_q = np.zeros(4)
        for name, pose in self.transforms.items():
            if self.listener.frameExists(name):
                t = self.listener.getLatestCommonTime(self.root, name)
                p, q = self.listener.lookupTransform(self.root, name, t)
                F = pm.fromTf((p, q))
                F = F * pose
                p, q = pm.toTf(F)
                avg_p += np.array(p)
                avg_q += np.array(q)
                count += 1
        if count > 0:
            avg_p /= count
            avg_q /= count
            avg_q /= np.linalg.norm(avg_q)
            self.broadcaster.sendTransform(avg_p, avg_q, rospy.Time.now(), self.name, self.root)

